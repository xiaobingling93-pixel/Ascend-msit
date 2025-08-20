#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ascend_utils.common.security import safe_copy_file
from msmodelslim.app.quant_service.base import BaseModelAdapter, BaseQuantConfig, BaseQuantService
from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.logging import set_logger_level
from .api import process_model
from .quant_config import ModelslimV1QuantConfig


def get_padding_data(tokenizer, calib_list, device_type):
    """
    Get the padding data for the calibration.
    """
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
    """
    Get the batch tokenized data for the calibration.
    """
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


def copy_files(input_path, output_path):
    """
    复制模型配置文件
    @param input_path: 源目录
    @param output_path: 目标目录
    """
    for file in os.listdir(input_path):
        if not any((file.endswith(subfix) for subfix in ['.json', '.py'])):
            continue

        if any((file.endswith(subfix) for subfix in ['index.json'])):
            continue

        ori_file = os.path.join(input_path, file)
        dest_file = os.path.join(output_path, file)
        safe_copy_file(src_path=ori_file, dest_path=dest_file)
        os.chmod(dest_file, int("600", 8))


class ModelslimV1QuantService(BaseQuantService):
    backend_name: str = "modelslim_v1"

    def __init__(self, dataset_loader: DatasetLoaderInterface):
        super().__init__(dataset_loader)

    def quantize(self, model: BaseModelAdapter, quant_config: BaseQuantConfig, save_path: Optional[Path] = None):
        if not isinstance(model, BaseModelAdapter):
            raise ValueError("model must be a BaseModelAdapter")
        if not isinstance(quant_config, BaseQuantConfig):
            raise ValueError("task must be a BaseTask")
        if save_path is not None and not isinstance(save_path, Path):
            raise ValueError("save_path must be a Path or None")

        return self.quant_process(model, ModelslimV1QuantConfig.from_base(quant_config), save_path)

    def quant_process(self, model: BaseModelAdapter, quant_config: ModelslimV1QuantConfig, save_path: Optional[Path]):

        set_logger_level("info")

        get_logger().info(f"==========QUANTIZATION: Prepare Dataset==========")
        dataset = self.dataset_loader.get_dataset_by_name(quant_config.spec.dataset)
        calib_data = get_tokenized_data(model.tokenizer, dataset, device=model.device.value)

        for save_cfg in quant_config.spec.save:
            save_cfg.set_save_directory(save_path)

        final_process_cfg = quant_config.spec.process + quant_config.spec.save

        process_model(model.model, final_process_cfg, calib_data)

        model.persisted(save_path)

        copy_files(str(model.ori), str(model.path))

        get_logger().info(f"quantized model: \n {model.model}")
        get_logger().info(f"==========QUANTIZATION: END==========")
