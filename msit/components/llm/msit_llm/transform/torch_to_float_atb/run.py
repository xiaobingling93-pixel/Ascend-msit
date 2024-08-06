# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import copy
import json
import math
import os
import time
from typing import Dict, List
from typing import Optional
import importlib
import sys

import torch
import torch_npu

from transformers.configuration_utils import PretrainedConfig

from atb_llm.utils import file_utils
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils import bind_cpus, initialize_distributed, Weights
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from examples.server.cache import CacheConfig, ModelConfig, CacheManager
from examples.server.generate import decode_token, generate_req
from examples.server.request import request_from_token


def get_model(model_name_or_path: str,
              max_position_embeddings: Optional[int] = None,
              is_flash_causal_lm: bool = True,
              revision: Optional[str] = None,
              trust_remote_code: bool = True,
              transformed_model_path: str = '.'):
    if file_utils.path_exists(model_name_or_path):
        model_name_or_path = file_utils.path_check(model_name_or_path)
    file_utils.check_owner(model_name_or_path)

    config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    model_type = config_dict['model_type'].lower()
    if model_type == "kclgpt":
        model_type = "codeshell"

    sys.path.insert(0, transformed_model_path)
    router_path = f"router_{model_type}"
    router = importlib.import_module(router_path)
    router_cls = getattr(router, f"{model_type.capitalize()}Router")
    router_ins = router_cls(
        model_name_or_path,
        max_position_embeddings,
        is_flash_causal_lm,
        revision,
        trust_remote_code,
        config_dict)
    return router_ins


class ModelRunner:
    model = None,
    soc_info = None,
    head_size = None,
    num_heads = None,
    num_kv_heads = None,
    num_layers = None
    device = None,
    dtype = None,

    def __init__(self, model_name_or_path, rank, world_size,
                 npu_id=None,
                 local_rank=None,
                 kv_cache_dtype=None,
                 max_position_embeddings=None,
                 is_flash_causal_lm: bool = True,
                 transformed_model_path='',
                 ):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.local_rank = local_rank if local_rank is not None else rank
        self.npu_id = npu_id if npu_id is not None else self.local_rank
        self.world_size = world_size

        if ENV.bind_cpu:
            try:
                bind_cpus(world_size, self.npu_id, ratio=1.0)
            except RuntimeError as e:
                print_log(rank, logger.info, e)
            except Exception as _:
                print_log(rank, logger.info, 'Skip binding cpu.')
        router_ins = get_model(model_name_or_path, max_position_embeddings, is_flash_causal_lm,
                               revision=None, trust_remote_code=True, transformed_model_path=transformed_model_path)
        self.model_cls = router_ins.model_cls
        self.config = router_ins.config
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.postprocessor = router_ins.postprocessor

        self.quantize = self.config.quantize
        self.kv_quant = self.config.kv_quant
        self.dtype = self.config.torch_dtype

        print_log(rank, logger.info, f'model_runner.quantize: {self.quantize}\n, '
                                     f'model_runner.kv_quant: {self.kv_quant}\n, '
                                     f'model_runner.dytpe: {self.dtype}')

        if self.kv_quant is not None and self.kv_quant not in ['C8']:
            raise NotImplementedError(
                f'unsupported type: {self.kv_quant}, 此类型从权重文件config.json中的`kv_quant`字段中获取；'
                f'若config.json中不存在此字段，请新增；当前此字段仅接受`C8`一种类型，'
                f'各模型具体支持的类型不同，请参考模型README文件。'
            )

        if self.dtype not in [torch.float16, torch.bfloat16]:
            raise NotImplementedError(
                f'unsupported type: {self.dtype}, 此类型从权重文件config.json中的`torch_dtype`字段中获取；'
                f'若config.json中不存在此字段，请新增；当前此字段仅接受`float16`和`bfloat16`两种类型，'
                f'各模型具体支持的类型不同，请参考模型README文件。'
            )

        self.process_group, self.device = initialize_distributed(self.rank, self.npu_id, world_size)
        torch.npu.set_compile_mode(jit_compile=False)

        print_log(rank, logger.info, f'init tokenizer done: {self.tokenizer}')

    def load_weights(self):
        weights = Weights(
            self.model_name_or_path, self.device, self.dtype,
            process_group=self.process_group,
            quantize=self.quantize,
            revision=None,
            extension=".safetensors"
        )
        self.model = self.model_cls(self.config, weights)

        self.model.to(weights.device)

        self.soc_info = self.model.soc_info
        self.head_size = self.model.head_size
        self.num_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.num_layers = self.model.num_layers

        print_log(self.rank, logger.info, f'model:\n {self.model}')

    def build_inputs(self, conversations: List[List[Dict[str, str]]], **kwargs):
        return [self.input_builder.make_context(self.rank, conversation, **kwargs) for conversation in conversations]

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def save_pretrained(self, **kwargs):
        save_directory_key = 'save_directory'
        if save_directory_key not in kwargs:
            raise ValueError(f'{save_directory_key} is required')
        kwargs[save_directory_key] = os.path.join(kwargs[save_directory_key], f'part{self.rank}-of-{self.world_size}')
        return self.model.save_pretrained(**kwargs)


class PARunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.world_size = kwargs.get('world_size', '1')

        self.model_path = kwargs.get('model_path', None)
        self.input_text = kwargs.get('input_text', None)

        self.max_batch_size = kwargs.get('max_batch_size', None)
        self.max_input_length = kwargs.get('max_input_length', None)
        self.max_output_length = kwargs.get('max_output_length', None)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', None)
        self.max_prefill_tokens = kwargs.get('max_prefill_tokens', None)

        self.block_size = kwargs.get('block_size', None)
        self.chat_template = kwargs.get('chat_template', None)
        self.is_flash_model = kwargs.get('is_flash_model', None)

        self.model = ModelRunner(
            self.model_path, rank=self.rank, world_size=self.world_size,
            local_rank=self.local_rank,
            max_position_embeddings=self.max_position_embeddings,
            transformed_model_path=kwargs.get('transformed_model_path', '')
        )
        self.tokenizer = self.model.tokenizer
        if self.chat_template:
            self.tokenizer.chat_template = self._load_chat_template(self.chat_template)
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.kv_quant = self.model.kv_quant
        self.model.load_weights()

        self.device = self.model.device
        self.model_config = ModelConfig(self.model.num_heads,
                                        self.model.num_kv_heads,
                                        self.model.head_size,
                                        self.model.num_layers,
                                        self.model.device,
                                        self.model.dtype,
                                        self.model.soc_info,
                                        self.kv_quant)

        self.max_memory = NpuHbmInfo.get_hbm_capacity(self.local_rank, self.world_size, self.model.soc_info.need_nz)
        self.init_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'hbm_capacity(GB): {self.max_memory / (1024 ** 3)}, '
                                          f'init_memory(GB): {self.init_memory / (1024 ** 3)}')

        self.warm_up_memory = 0
        self.warm_up_num_blocks = 0
        self.cache_manager = None
        self.compress_head_enable = ENV.compress_head_enable

    def __repr__(self):
        return (
                "PARunner("
                + f"model_path={self.model_path}, "
                + f"input_text={self.input_text}, "
                + f"max_position_embeddings={self.max_position_embeddings}, "
                + f"max_input_length={self.max_input_length}, "
                + f"max_output_length={self.max_output_length}, "
                + f"max_prefill_tokens={self.max_prefill_tokens}, "
                + f"is_flash_model={self.is_flash_model}, "
                + f"max_batch_size={self.max_batch_size}, "
                + f"dtype={self.dtype}, "
                + f"block_size={self.block_size}, "
                + f"model_config={self.model_config}, "
                + f"max_memory={self.max_memory}, "
        )

    @staticmethod
    def _load_chat_template(chat_template: str):
        if os.path.exists(chat_template):
            with open(chat_template, "r", encoding="utf-8") as f:
                chat_template_content = f.read()
        else:
            chat_template_content = chat_template
        if chat_template_content:
            print_log(int(os.getenv("RANK", "0")), logger.info, f"Using chat template:\n{chat_template_content}")
        return chat_template_content

    def warm_up(self):
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
        all_input_length = self.max_batch_size * self.max_input_length
        input_ids = torch.ones(all_input_length, dtype=torch.int64).to(self.device)
        position_ids = torch.arange(self.max_input_length, dtype=torch.int32).repeat(self.max_batch_size).to(
            self.device)
        cu_seqlen_prefill = torch.tensor([1])
        try:
            block_num = math.ceil(all_input_length / self.block_size)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        block_tables_tensor = torch.arange(block_num, dtype=torch.int32).view(1, -1).to(self.device)
        slots = torch.arange(all_input_length, dtype=torch.int32).to(self.device)
        input_lengths_tensor = torch.tensor(
            [self.max_input_length] * self.max_batch_size, dtype=torch.int64
        ).to(self.device)
        prefill_head_indices = torch.tensor([all_input_length - 1], dtype=torch.int64).to(self.device)
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = math.ceil((self.max_input_length + self.max_output_length) /
                                                self.block_size) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        if self.compress_head_enable:
            cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size, \
                self.max_input_length, self.max_output_length, self.max_batch_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            is_prefill=cu_seqlen_prefill is not None,
            block_tables=block_tables_tensor,
            kv_cache=self.cache_manager.kv_cache,
            slots=slots,
            input_lengths=input_lengths_tensor,
            max_seq_len=self.max_input_length,
            lm_head_indices=prefill_head_indices
        )
        self.warm_up_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}')
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(self, inputs, batch_size, max_output_length, ignore_eos, is_chat_model=False, **kwargs):
        print_log(self.rank, logger.info, "---------------begin inference---------------")
        if ignore_eos:
            self.model.postprocessor.eos_token_id = []
        is_truncation = kwargs.get("truncation", False)
        input_ids = self._build_model_inputs(inputs, is_chat_model, is_truncation)
        if len(input_ids) == 1:
            req_list = [request_from_token(input_ids[0], max_output_length, self.block_size, req_idx=idx)
                        for idx in range(batch_size)]
        else:
            req_list = [request_from_token(input_ids_ins, max_output_length, self.block_size, req_idx=idx)
                        for idx, input_ids_ins in enumerate(input_ids)]
        print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')

        if not self.cache_manager:
            if self.max_prefill_tokens == -1:
                self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
            cache_block_size = self.block_size * self.model.num_kv_heads * self.model.head_size
            dtype_size = CacheManager.get_dtype_size(self.dtype)
            total_cache_size = self.model.num_layers * cache_block_size * 2 * dtype_size

            max_memory = ENV.memory_fraction * self.max_memory \
                if not ENV.max_memory_gb else int(ENV.max_memory_gb) * (1 << 30)
            free_memory = max_memory - ENV.reserved_memory_gb * (1 << 30) - (
                self.warm_up_memory if self.warm_up_memory != 0 else self.init_memory)
            print_log(self.rank, logger.info,
                      f"infer max_memory(GB): {max_memory / (1024 ** 3): .2f}, "
                      f"warm_up_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}, "
                      f"free_memory(GB): {free_memory / (1024 ** 3): .2f}")

            num_blocks = int(free_memory // total_cache_size)
            print_log(self.rank, logger.info, f"num_blocks: {num_blocks}, free_memory: {free_memory}")
            cache_config = CacheConfig(num_blocks, self.block_size)
            if self.compress_head_enable:
                cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size, \
                    self.max_input_length, self.max_output_length, self.max_batch_size)
            self.cache_manager = CacheManager(cache_config, self.model_config)

        if ENV.benchmark_enable:
            req_list_dummy = copy.deepcopy(req_list)
            self.model.postprocessor.max_new_tokens = 2
            generate_req(req_list_dummy, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)

        self.model.postprocessor.max_new_tokens = max_output_length
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        if not ENV.profiling_enable:
            print_log(self.rank, logger.debug, "no profiling")
            torch.npu.synchronize()
            e2e_start = time.time()
            generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
            _, _ = decode_token(req_list, self.tokenizer, skip_special_tokens)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
        else:
            print_log(self.rank, logger.debug, "enter profiling")
            profiling_path = ENV.profiling_filepath
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            profiler_level = torch_npu.profiler.ProfilerLevel
            target_level = "Level" + ENV.profiling_level
            if not hasattr(profiler_level, target_level):
                raise NotImplementedError(f"target_level: {target_level} is not implemented"
                                          f" in torch_npu.profiler.ProfilerLevel")
            actual_profiler_level = getattr(profiler_level, target_level)
            torch.npu.synchronize()
            e2e_start = time.time()
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=actual_profiler_level,
                l2_cache=False,
                data_simplification=False
            )
            with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_path),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config):
                generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start

        generate_text_list, token_num_list = decode_token(req_list, self.tokenizer, skip_special_tokens)
        if ENV.token_ids_save_enable:
            if self.local_rank == 0:
                for idx, req in enumerate(req_list):
                    input_ids_save_filename = f"input_ids_{idx}.pth"
                    output_ids_save_filename = f"output_ids_{idx}.txt"
                    torch.save(req.input_ids.cpu(),
                               os.path.join(ENV.token_ids_save_folder, input_ids_save_filename))
                    output_path = os.path.join(ENV.token_ids_save_folder, output_ids_save_filename)
                    with safe_open(output_path, "w", encoding='utf-8') as f:
                        f.write(' '.join(map(str, req.out_token_list)))
        print_log(self.rank, logger.info, "---------------end inference---------------")
        return generate_text_list, token_num_list, e2e_time

    def _build_model_inputs(self, inputs, is_chat_model, is_truncation=False):
        input_texts, input_ids, input_conversations = [], [], []
        if isinstance(inputs, list) and inputs:
            if isinstance(inputs[0], str):
                input_texts = inputs
            elif isinstance(inputs[0], torch.Tensor):
                input_ids = inputs
            elif isinstance(inputs[0], list) and inputs[0]:
                if isinstance(inputs[0][0], int):
                    input_ids = inputs
                elif isinstance(inputs[0][0], dict):
                    input_conversations = inputs
        if not (input_texts or input_ids or input_conversations):
            raise ValueError(f"The inputs of `PARunner.infer` must be as List[str], List[torch.Tensor], List[List[int]]"
                             f" or List[List[Dict]]. Now the inputs ({inputs}) is not acceptable or is empty.")
        if is_chat_model:
            if input_conversations:
                input_ids = self.model.build_inputs(input_conversations)
            elif input_texts:
                input_conversations = [[{"role": "user", "content": t}] for t in input_texts]
                input_ids = self.model.build_inputs(input_conversations)
            else:
                print_log(self.rank, logger.warning, "Neither conversations nor input_texts exist, "
                                                     "'chat' parameter is not effective.")
        elif input_texts:
            input_ids = [self.tokenizer([text], return_tensors="pt", truncation=is_truncation)["input_ids"].flatten()
                for text in input_texts]
        return input_ids


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be a boolean")


def parse_ids(list_str):
    return [int(item) for item in list_str.split(',')]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help="model and tokenizer path", default='/data/chatglm2_6b')
    parser.add_argument('--transformed_model_path', help="transformed model path", default='./chatglm')    
    parser.add_argument(
        '--input_texts',
        type=str,
        nargs='+',
        default=["What's deep learning?"])
    parser.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None)
    parser.add_argument(
        '--input_file',
        type=str,
        help='CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)

    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--max_position_embeddings', type=int, default=None)
    parser.add_argument('--max_prefill_tokens', type=int, default=-1)

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument('--chat_template', type=str, default=None)
    parser.add_argument('--ignore_eos', action='store_true')
    parser.add_argument('--is_chat_model', action='store_true')
    parser.add_argument('--is_flash_model', action='store_false')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }

    if args.input_ids:
        infer_inputs = args.input_ids
    else:
        infer_inputs = args.input_texts
    if args.is_chat_model and args.input_file:
        conversations = []
        with open(args.input_file, 'r', encoding='utf-8') as file:
            for line in file:
                data_line = json.loads(line)
                conversations.append(data_line)
        infer_inputs = conversations

    pa_runner = PARunner(**input_dict)
    print_log(rank, logger.info, f'pa_runner: {pa_runner}')
    pa_runner.warm_up()

    infer_params = {
        "inputs": infer_inputs,
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
        "is_chat_model": args.is_chat_model
    }
    generate_texts, token_nums, _ = pa_runner.infer(**infer_params)

    length = len(infer_inputs)
    for i, generate_text in enumerate(generate_texts):
        if i < length:
            print_log(rank, logger.info, f'Question[{i}]: {infer_inputs[i]}')
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
