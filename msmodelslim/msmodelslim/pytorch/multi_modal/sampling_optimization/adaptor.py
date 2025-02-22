# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
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
from .model_open_sora_plan1_2_sp import ReStepOpenSoraPipelineV1_2

logger = logging.getLogger(__name__)


def dump_json(data, filename="output.json", permissions=0o640):
    """
    Dump JSON data to a file with specified file permissions.

    This function creates the file using os.open to set permissions,
    taking into account the system umask, and then confirms the file
    permissions with os.chmod.

    Parameters:
        data: JSON-serializable data.
        filename (str): Output file path (default "output.json").
        permissions (int): File permissions (default 0o650, i.e. owner read/write, group read/execute).

    Raises:
        Exception: Propagates any error encountered during file writing or permission setting.
    """
    try:
        # 使用 os.open 创建文件，指定权限，并确保存在时覆盖（O_TRUNC）
        with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, permissions), 'w') as f:
            json.dump(data, f, indent=4)

        # 确认文件权限
        os.chmod(filename, permissions)
    except Exception as e:
        logger.error("Error saving JSON data", exc_info=True)
        raise


# ----------------- ReStep 相关数据结构 -----------------
@dataclass
class ReStepSearchConfig:
    videos_path: str = None
    save_dir: str = None
    neighbour_type: str = 'uniform'
    monte_carlo_iters: int = 5

    num_sampling_steps: int = 50


def check_exist_and_read_permission(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"Read permission is denied for the path: {path}")


def check_exist_and_write_permission(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    if not os.access(path, os.W_OK):
        raise PermissionError(f"Write permission is denied for the path: {path}")


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
        self.videos_paths = None

        if not isinstance(pipeline, OpenSoraPipeline):
            raise ValueError("pipeline must be OpenSoraPipeline")

        if not isinstance(config, ReStepSearchConfig):
            raise ValueError("config must be ReStepSearchConfig")

        self.check_search_config(config)

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = torch.cuda.current_device()
        else:
            raise RuntimeError("RANK and WORLD_SIZE must be set in environment")

        pipeline: ReStepOpenSoraPipelineV1_2 \
            = self.replace_obj_class(pipeline, ReStepOpenSoraPipelineV1_2)

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

    def check_search_config(self, config: ReStepSearchConfig) -> None:
        """
        Validate the search configuration.

        Checks performed:
          - Ensure `config.videos_path` exists and has both read and write permissions.
          - Verify `config.num_sampling_steps` is an integer and at least 1.
          - Confirm `config.neighbour_type` is either 'uniform' or 'random'.
          - Ensure the videos directory contains between 5 and 20 .mp4 files.
          - Validate `config.monte_carlo_iters` is an integer greater than 0 and
            less than or equal to the number of .mp4 videos in the directory.

        Raises:
            ValueError: If any of the validations fail.
        """
        # Check that the videos_path exists with the appropriate permissions.
        check_exist_and_read_permission(config.videos_path)
        check_exist_and_write_permission(config.videos_path)

        # Validate the number of sampling steps.
        if not isinstance(config.num_sampling_steps, int) or config.num_sampling_steps < 1:
            raise ValueError("config.num_sampling_steps must be an integer and >= 1")

        # Validate the neighbour type.
        if config.neighbour_type not in {'uniform', 'random'}:
            raise ValueError("config.neighbour_type must be either 'uniform' or 'random'")

        # Gather all .mp4 files in the specified videos_path.
        video_dir = config.videos_path
        mp4_files = glob.glob(os.path.join(video_dir, '*.mp4'))
        num_videos = len(mp4_files)

        if not (5 <= num_videos <= 20):
            raise ValueError("videos_path must contain between 5 and 20 .mp4 videos")

        # Validate monte_carlo_iters based on the number of available videos.
        if not (isinstance(config.monte_carlo_iters, int) and 0 < config.monte_carlo_iters <= num_videos):
            raise ValueError(
                "config.monte_carlo_iters must be an integer greater than 0 "
                "and <= the number of .mp4 videos in videos_path")

        self.videos_paths = mp4_files

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

        videos_paths = self.videos_paths

        save_dir = config.save_dir

        os.makedirs(save_dir, exist_ok=True)
        save_file_path = os.path.join(save_dir, 'searched_schedule.txt')
        logger.info("Result will saved at: %s", save_file_path)

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

        dump_json(schedule, save_file_path)
        logger.info("Search result saved at: %s", save_file_path)

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
