import torch.distributed

try:
    from opensora.sample.pipeline_opensora_sp import *
except ImportError:
    raise ImportError("Cannot find package Open-Sora-Plan 1.2, please install it first.")

from typing import Union


class ReStep_OpenSoraPipeline_v_1_2(OpenSoraPipeline):
    def __init__(self, tokenizer: T5Tokenizer, text_encoder: T5EncoderModel, vae: AutoencoderKL,
                 transformer: Transformer2DModel, scheduler: DPMSolverMultistepScheduler):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        self.args = None

    def one_step_sample(self, latents, timestep, step_index, encoder_states, extra_step_kwargs, added_cond_kwargs, ):
        timestep *= self.scheduler.num_train_timesteps

        timesteps_old = self.scheduler.timesteps.cpu().numpy().tolist()
        timesteps = timesteps_old.copy()
        timesteps[step_index] = timestep

        self.scheduler.set_timesteps(len(timesteps), timesteps, device=latents.device)
        latents = self.scheduler.scale_model_input(latents, timestep)

        bs = latents.shape[0]
        current_timestep = torch.tensor([timestep] * bs).to(latents.device)

        if get_sequence_parallel_state():
            latents = self.split_sequence(latents, hccl_info.rank, hccl_info.world_size)

        attention_mask = torch.ones_like(latents)[:, 0]
        if get_sequence_parallel_state():
            attention_mask = attention_mask.repeat(1, hccl_info.world_size, 1, 1)

        noise_pred = self.transformer(
            latents,
            attention_mask=attention_mask,
            timestep=current_timestep,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            **encoder_states
        )[0]

        latent_channels = self.transformer.config.in_channels
        if self.transformer.config.out_channels // 2 == latent_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        else:
            noise_pred = noise_pred

        # compute previous image: x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
        self.scheduler.set_timesteps(len(timesteps), timesteps_old, device=latents.device)

        if get_sequence_parallel_state():
            latents = self.gather_sequences(latents, hccl_info.world_size)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        return latents

    @staticmethod
    def split_sequence(sequence, local_rank, world_size):
        old_shape = sequence.shape
        x = sequence.shape[2] // world_size
        sequence = rearrange(sequence.view((*old_shape[:3], -1)), 'b c (n x) s -> b c n x s', n=world_size,
                             x=x).contiguous()
        sequence = sequence[:, :, local_rank, :, :]
        return sequence.view((*old_shape[:2], x, *old_shape[3:]))

    @staticmethod
    def gather_sequences(sequence, world_size):
        sequence_shape = list(sequence.shape)
        full_shape = [sequence_shape[0] * world_size] + sequence_shape[1:]
        all_sequences = torch.zeros(full_shape, dtype=sequence.dtype, device=sequence.device)
        torch.distributed.all_gather_into_tensor(all_sequences, sequence)
        sequences_list = list(all_sequences.chunk(world_size, dim=0))
        sequence = torch.cat(sequences_list, dim=2)
        return sequence

    @staticmethod
    def get_sequence_parallel_state():
        return get_sequence_parallel_state()

    @torch.no_grad()
    def get_text_embeddings(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: str = "",
            guidance_scale: float = 4.5,
            num_images_per_prompt: Optional[int] = 1,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            prompt_attention_mask: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
            clean_caption: bool = True,
            max_sequence_length: int = 300,
            **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:

        # import ipdb;ipdb.set_trace()
        device = getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')
        device = kwargs.get('device', device)  # fix bug

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )
