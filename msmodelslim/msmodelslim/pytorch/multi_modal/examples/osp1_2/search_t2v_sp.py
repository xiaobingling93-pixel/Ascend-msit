import os
import argparse

import torch
import torch.distributed as dist
import torch_npu
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, MT5EncoderModel

from opensora.models.causalvideovae import ae_stride_config, CausalVAEModelWrapper
from opensora.models.diffusion.udit.modeling_udit import UDiTT2V
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from opensora.npu_config import npu_config
from opensora.acceleration.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state
from opensora.sample.pipeline_opensora_sp import OpenSoraPipeline

from msmodelslim.pytorch.multi_modal.sampling_optimization.scheduler import EulerAncestralDiscreteSchedulerExample


def load_t2v_checkpoint(model_path):
    print('load_t2v_checkpoint, ', model_path)
    if args.model_type == 'udit':
        transformer_model = UDiTT2V.from_pretrained(model_path, cache_dir=args.cache_dir,
                                                    low_cpu_mem_usage=False, device_map=None,
                                                    torch_dtype=weight_dtype)
    elif args.model_type == 'dit':
        transformer_model = OpenSoraT2V.from_pretrained(model_path, cache_dir=args.cache_dir,
                                                        low_cpu_mem_usage=False, device_map=None,
                                                        torch_dtype=weight_dtype)
    else:
        transformer_model = LatteT2V.from_pretrained(model_path, cache_dir=args.cache_dir, low_cpu_mem_usage=False,
                                                     device_map=None, torch_dtype=weight_dtype)

    # set eval mode
    transformer_model.eval()

    pipeline = OpenSoraPipeline(vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                transformer=transformer_model).to(device)

    if args.compile:
        pipeline.transformer = torch.compile(pipeline.transformer)

    return pipeline


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--model_type', type=str, default="dit", choices=['dit', 'udit', 'latte'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true')

    # sample_optimization
    parser.add_argument("--save_dir", type=str, required=True,
                        help='The path to save file about the optimized schedule timestep')
    parser.add_argument("--videos_path", type=str, required=True, help='The path of calibration videos')
    parser.add_argument("--neighbour_type", type=str, required=False, default="uniform")
    parser.add_argument("--monte_carlo_iters", type=int, required=False, default=5)

    args = parser.parse_args()

    if torch_npu is not None:
        npu_config.print_msg(args)

    # 初始化分布式环境
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size)
    if torch_npu is not None and npu_config.on_npu:
        torch_npu.npu.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)
    weight_dtype = torch.bfloat16
    device = f"npu:{torch.cuda.current_device()}"
    vae = CausalVAEModelWrapper(args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_sample_min_size = 512
        vae.vae.tile_latent_min_size = 64
        vae.vae.tile_sample_min_size_t = 29
        vae.vae.tile_latent_min_size_t = 8
        if args.save_memory:
            vae.vae.tile_sample_min_size = 256
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8

    vae.vae_scale_factor = ae_stride_config[args.ae]

    text_encoder = MT5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir,
                                                   low_cpu_mem_usage=True, torch_dtype=weight_dtype).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)

    # set eval mode
    vae.eval()
    # text_encoder.eval()
    text_encoder.bfloat16().eval()

    if args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteSchedulerExample()
    else:
        raise ValueError(f'args.sample_method: {args.sample_method} not supported.')

    if args.num_frames not in {29}:
        raise ValueError(f'args.num_frames: {args.sample_method} not supported. Only support 29 frames.')

    # read text_prompt
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]

    save_img_path = args.save_img_path
    full_path = args.model_path

    from msmodelslim.pytorch.multi_modal.sampling_optimization import ReStepSearchConfig, ReStepAdaptor

    # get original pipeline
    pipeline: OpenSoraPipeline = load_t2v_checkpoint(full_path)
    # save osp1.2 config args to the pipeline obj
    pipeline.args = args

    # set restep search config
    config = ReStepSearchConfig(
        videos_path=args.videos_path,
        save_dir=args.save_dir,
        neighbour_type=args.neighbour_type,
        monte_carlo_iters=args.monte_carlo_iters,
    )

    # create ReStepAdaptor
    restep_adaptor = ReStepAdaptor(pipeline, config)

    # do the scheduler timestep search
    scheduler_timestep = restep_adaptor.search()
