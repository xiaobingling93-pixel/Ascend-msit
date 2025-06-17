# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import functools
import os
from typing import List

from tqdm import tqdm
import torch
import torch.nn.functional as F

from ascend_utils.common.security import get_valid_read_path
from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.tools.logger import set_logger_level
from msmodelslim.tools.convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16, OpsType
from msmodelslim.tools.add_safetensors import add_safetensors
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.app.naive_quantization.practice_data import ConfigTask
from msmodelslim.utils.safe_utils import SafeGenerator


class HookRegistry:
    def __init__(self):
        """Registering a special process for a model"""
        self.functions = {}

    def register(self, hook_name, function_name, func):
        self.functions.setdefault(hook_name, {})[function_name] = func

    def get(self, hook_name, function_name):
        return self.functions.get(hook_name, {}).get(function_name, None)


def custom_hook(model_config):
    model_config["mla_quantize"] = "w8a8"
    model_config["quantize"] = "w8a8_dynamic"
    model_config["moe_quantize"] = "w4a8_dynamic"
    model_config["model_type"] = "deepseekv2"


def get_padding_data(tokenizer, calib_list, device_type):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt', add_special_tokens=False)
        calib_dataset.append(
            inputs.data['input_ids'].to(device_type)
        )
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    new_calib_dataset = []
    for inputs in calib_dataset:
        new_inputs = F.pad(inputs, (0, max_len - inputs.size(1)), value=0)
        new_calib_dataset.append(new_inputs)
    return [torch.cat(new_calib_dataset)]


def get_batch_tokenized_data(model_tokenizer, calib_list, batch_size, device="npu"):
    calib_dataset = []
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        tmp = get_padding_data(model_tokenizer, calib_data, device)
        calib_dataset.append(tmp)
    return calib_dataset


def get_tokenized_data(tokenizer, calib_list, device,
                       input_ids_name='input_ids',
                       attention_mask_name='attention_mask'):
    tokenized_data = []
    for input_text in calib_list:
        inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)
        tokenized_data.append(
            [inputs.data[input_ids_name], inputs.data[attention_mask_name]])
    return tokenized_data


def convert_model(model, model_path):
    auto_convert_model_fp8_to_bf16(model, model_path, OpsType.AUTO)
    return model


def add_safetensors_func(model_path, save_path):
    add_safetensors(org_paths=model_path, target_dir=save_path, safetensors_prefix="mtp_float",
                    max_file_size_gb=5, prefix="model.layers.61.")


class Quantization:
    def __init__(self):
        self.hook_registry = HookRegistry()
        self.hook_registry.register("convert_dtype", "deepseekv2", convert_model)
        self.hook_registry.register("post_quantization", "deepseekv2", add_safetensors_func)
        self.hook_registry.register("customized_hook_ds", "deepseekv2", custom_hook)

    def quant_process(self, config_task: ConfigTask):
        set_logger_level("info")

        # handle params
        customized_params = config_task.customized_config
        if customized_params is not None:
            model_path = customized_params.model_path
            save_path = customized_params.save_path
            device_type = customized_params.device
            trust_remote_code = customized_params.trust_remote_code
        else:
            raise ValueError("Required parameters are missing.")

        tokenizer_cfg = config_task.specific.tokenizer_cfg
        model_cfg = config_task.specific.model_cfg

        anti_cfg = None
        tmp = config_task.specific.anti_cfg
        if tmp is not None:
            anti_cfg = AntiOutlierConfig(dev_type=device_type, **tmp)

        use_fa_quant = bool(config_task.specific.calib_cfg.pop('use_fa_quant', False))
        fa_amp = config_task.specific.calib_cfg.pop('fa_amp', 0)
        
        anti_params = config_task.specific.anti_params
        calib_cfg = QuantConfig(dev_type=device_type, **config_task.specific.calib_cfg)
        if use_fa_quant:
            calib_cfg = calib_cfg.fa_quant(fa_amp)
        calib_params = config_task.specific.calib_params
        calib_save_params = config_task.specific.calib_save_params

        pbar = tqdm(total=5, position=0, desc="Total Process")

        # load model and tokenizer
        safe_generator = SafeGenerator()
        auto_config = safe_generator.get_config_from_pretrained(model_path, **model_cfg)
        device_map = 'cpu' if device_type == 'cpu' else 'auto'
        dtype = auto_config.torch_dtype if device_type == 'npu' else torch.float32

        tokenizer = safe_generator.get_tokenizer_from_pretrained(model_path, trust_remote_code=trust_remote_code, **tokenizer_cfg)
        model = safe_generator.get_model_from_pretrained(model_path=model_path, device_map=device_map,
                                                         torch_dtype=dtype, trust_remote_code=trust_remote_code, **model_cfg)

        convert_dtype = self.hook_registry.get("convert_dtype", auto_config.model_type)
        if convert_dtype is not None:
            model = convert_dtype(model, model_path)

        pbar.update(1)

        # handle dataset        
        anti_path = config_task.specific.anti_file
        calib_path = config_task.specific.calib_file
        batch_size = config_task.specific.batch_size
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        calib_data = None

        if calib_path is not None:
            calib_path = os.path.abspath(os.path.join(cur_dir, calib_path))
            calib_path = get_valid_read_path(calib_path, "jsonl", is_dir=False)
            calib_prompt = safe_generator.load_jsonl(calib_path)
            calib_data = get_tokenized_data(tokenizer, calib_prompt, device=model.device)

        anti_data = calib_data

        if anti_path is not None:
            anti_path = os.path.abspath(os.path.join(cur_dir, anti_path))
            anti_path = get_valid_read_path(anti_path, "jsonl", is_dir=False)
            anti_prompt = safe_generator.load_jsonl(anti_path)
            anti_data = get_batch_tokenized_data(tokenizer, anti_prompt, batch_size, device=model.device)

        if anti_cfg is not None:
            anti_outlier = AntiOutlier(model=model, calib_data=anti_data, cfg=anti_cfg, **anti_params)
            anti_outlier.process()

        pbar.update(1)

        if device_type == "npu":
            # 如果使用npu进行量化需开启二进制编译，避免在线编译算子
            torch.npu.set_compile_mode(jit_compile=False)
            
        # quantization
        calibrator = Calibrator(model=model, cfg=calib_cfg, calib_data=calib_data, **calib_params)
        calibrator.run()

        pbar.update(1)

        calibrator.save(output_path=save_path, **calib_save_params)

        # handle config
        customized_hook_ds = self.hook_registry.get("customized_hook_ds", auto_config.model_type)
        if customized_hook_ds is not None:
            custom_hooks = {
                'config.json': functools.partial(modify_config_json, custom_hook=customized_hook_ds)
            }
            copy_config_files(input_path=model_path, output_path=save_path, quant_config=calib_cfg,
                              custom_hooks=custom_hooks)
        else:
            w_bit = calib_cfg.w_bit
            a_bit = calib_cfg.a_bit
            quant_type = f"w{w_bit}a{a_bit}"
            is_sparse_compress = w_bit == 4 and a_bit == 8 and (calib_cfg.co_sparse or calib_cfg.is_lowbit)
            if is_sparse_compress:
                quant_type = "w8a8s"
            is_w8a8_dynamic = w_bit == 8 and a_bit == 8 and calib_cfg.is_dynamic
            if is_w8a8_dynamic:
                quant_type = "w8a8_dynamic"
            if calib_cfg.model_quant_type == "W4A8_DYNAMIC":
                quant_type = "w4a8_dynamic"

            config_map = config_task.specific.calib_cfg
            if anti_cfg is not None:
                config_map.update(config_task.specific.anti_cfg)

            if device_type:
                config_map["dev_type"] = device_type
            safe_generator.modify_config(model_path, save_path, auto_config.torch_dtype,
                                         quant_type, **config_map)
            safe_generator.copy_tokenizer_files(model_path, save_path)

        pbar.update(1)

        # deepseek-v2 add safetensors
        post_quantization = self.hook_registry.get("post_quantization", auto_config.model_type)
        if post_quantization:
            post_quantization(model, model_path, save_path)
        pbar.update(1)
