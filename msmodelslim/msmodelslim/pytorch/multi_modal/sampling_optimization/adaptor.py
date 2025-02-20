import os
import glob
import random
import functools
from dataclasses import dataclass
import logging
import gc

import numpy as np
import torch

from opensora.sample.pipeline_opensora_sp import OpenSoraPipeline

from .schedule_optimizer import AYSOptimizer
from .model_open_sora_plan1_2_sp import ReStep_OpenSoraPipeline_v_1_2

logger = logging.getLogger(__name__)


# ----------------- ReStep 相关数据结构 -----------------
@dataclass
class ReStepSearchConfig:
    videos_path: str = None
    save_dir: str = None
    neighbour_type: str = 'uniform'
    monte_carlo_iters: int = 5

    num_sampling_steps: int = 50


# ----------------- 适配器实现 -----------------
class ReStepAdaptor:
    _timestep_idx = None

    def __init__(self, pipeline: OpenSoraPipeline,
                 config: ReStepSearchConfig,
                 ):
        """
        pipeline: 为官方代码的 OpenSoraPipeline 类型
        config: ReStepSearchConfig 配置对象
        """
        self.search_config = config

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = torch.cuda.current_device()
        else:
            raise RuntimeError("RANK and WORLD_SIZE must be set in environment")

        pipeline: ReStep_OpenSoraPipeline_v_1_2 \
            = self.replace_obj_class(pipeline, ReStep_OpenSoraPipeline_v_1_2)

        self.pipeline = pipeline

    @staticmethod
    def replace_obj_class(obj, new_obj_class):
        obj.__class__ = new_obj_class
        return obj

    @staticmethod
    def clear_cache():
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()

    def search(self):
        pipeline_args = self.pipeline.args

        config = self.search_config
        device = f"npu:{torch.cuda.current_device()}"
        pipeline = self.pipeline
        scheduler = self.pipeline.scheduler

        num_sampling_steps_set = getattr(pipeline_args, 'num_sampling_steps', None)
        if num_sampling_steps_set is not None and num_sampling_steps_set != config.num_sampling_steps:
            raise ValueError("pipeline_args.num_sampling_steps must be equal as {}".format(config.num_sampling_steps))

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.get_default_prompt(pipeline_args)

        bs = 1
        no_guid_states = {'encoder_hidden_states': negative_prompt_embeds.expand(bs, -1, -1).unsqueeze(1),
                          'encoder_attention_mask': negative_prompt_attention_mask.expand(bs, -1).unsqueeze(1)}
        if self.pipeline.get_sequence_parallel_state():
            no_guid_states['encoder_hidden_states'] = pipeline.split_sequence(no_guid_states['encoder_hidden_states'],
                                                                              self.local_rank, self.world_size)

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        generator = torch.Generator(device=device)
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator=generator, eta=None)

        vid_path = config.videos_path
        videos_paths = list(glob.glob(os.path.join(vid_path, '*'), recursive=False))
        save_dir = config.save_dir

        os.makedirs(save_dir, exist_ok=True)
        logger.debug("Result will saved at: %s", save_dir)

        denoising_fn = functools.partial(pipeline.one_step_sample,
                                         encoder_states=no_guid_states,
                                         extra_step_kwargs=extra_step_kwargs,
                                         added_cond_kwargs=added_cond_kwargs,
                                         )

        scheduler.set_timesteps(config.num_sampling_steps, device=device)

        optimizer = AYSOptimizer(denoising_fn, scheduler, pipeline.vae, videos_paths, device=device, batch_size=bs,
                                 neighbourhood_type=config.neighbour_type, save_dir=save_dir)
        schedule = [x / 1000 for x in scheduler.timesteps.tolist()][::-1]

        with torch.no_grad():
            schedule = optimizer.optimize(schedule, config.monte_carlo_iters)

        return schedule

    def seed_everything(self, seed):
        seed += self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_default_prompt(self, args):
        positive_prompt = """
            (masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
            {}. 
            emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
            sharp focus, high budget, cinemascope, moody, epic, gorgeous
            """

        negative_prompt = """
            nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
            low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
            """

        self.seed_everything(42)
        res = self.pipeline.get_text_embeddings(positive_prompt,
                                                negative_prompt=negative_prompt,
                                                num_frames=args.num_frames,
                                                height=args.height,
                                                width=args.width,
                                                num_inference_steps=args.num_sampling_steps,
                                                guidance_scale=args.guidance_scale,
                                                num_images_per_prompt=1,
                                                mask_feature=True,
                                                device=f"npu:{torch.cuda.current_device()}",
                                                max_sequence_length=args.max_sequence_length,
                                                )

        self.pipeline.text_encoder = self.pipeline.text_encoder.to('cpu')
        self.clear_cache()
        return res
