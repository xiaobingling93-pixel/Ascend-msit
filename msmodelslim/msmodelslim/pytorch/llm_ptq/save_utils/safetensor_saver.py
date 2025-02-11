# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from collections import OrderedDict
from typing import Dict

import torch

from safetensors.torch import save_file

from ascend_utils.common.security import json_safe_dump, SafeWriteUmask, check_type, get_valid_write_path, get_write_directory
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save_utils import get_index_json

from .tensor_collector import BaseSaver


class SafeTensorSaver(BaseSaver):
    GB_SIZE = 1 * 1024 * 1024 * 1024

    def __init__(self, max_gb_size: int = 4, save_directory: str = '.', save_prefix: str = "quant_model_weight"):
        super().__init__()
        self.saved_keys_map: Dict[str, str] = {}
        self.wait_save_keys: Dict[str, torch.Tensor] = {}
        self.max_size: int = max_gb_size * SafeTensorSaver.GB_SIZE
        self.model_quant_type: QuantType = self.cfg.model_quant_type
        self.save_prefix: str = f"{save_prefix}_{self.model_quant_type}"
        self.total_size: int = 0
        self._wait_save_size: int = 0
        self._save_count: int = 0
        self._save_directory: str = save_directory

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not self.is_enabled:
            return
        
        check_type(value, torch.Tensor)
        tensor = value.detach().cpu().contiguous()
        
        if tensor.device.type == 'meta':
            msmodelslim_logger.warning(f"Skip meta tensor {key}")
            return
        
        tensor_size = tensor.numel() * tensor.element_size()
        self.wait_save_keys[key] = tensor.cpu()
        self.total_size += tensor_size
        self._wait_save_size += tensor_size
        msmodelslim_logger.debug(f"Add new tensor {key}, device: {tensor.device}, size: {tensor_size}, total: {self._wait_save_size}")

        if self._wait_save_size >= self.max_size:
            self.save_one_file()

    @property
    def save_directory(self) -> str:
        return self._save_directory

    @save_directory.setter
    def save_directory(self, value: str) -> None:
        self._save_directory = get_write_directory(value, write_mode=0o750)


    def post_process(self) -> None:
        # save last file if needed
        if self.wait_save_keys:
            self.save_one_file()

        # rename safetensors
        for i in range(self._save_count):
            src_file = os.path.join(self.save_directory, f"{self.save_prefix}-{i + 1:05d}-of-00000.safetensors")
            dst_file = os.path.join(self.save_directory, f"{self.save_prefix}-{i + 1:05d}-of-{self._save_count:05d}.safetensors")
            msmodelslim_logger.info(f"{src_file} -> {dst_file}")
            shutil.move(src_file, dst_file)

        # process safetensor index json
        for key in self.saved_keys_map.keys():
            self.saved_keys_map[key] = self.saved_keys_map[key].removesuffix('-of-00000.safetensors') + f'-of-{self._save_count:05d}.safetensors'

        # save index json
        index_json_dict = get_index_json(self.saved_keys_map, self.total_size)
        index_json_name = os.path.join(self.save_directory, self.save_prefix + '.index.json')
        index_json_path = get_valid_write_path(index_json_name, extensions=[".json"])
        json_safe_dump(index_json_dict, index_json_path, indent=2)


    def save_one_file(self) -> None:
        self.__save_count += 1
        save_file_name = f"{self.save_prefix}-{self._save_count:05d}-of-00000.safetensors"
        full_save_file_name = os.path.join(self.save_directory, save_file_name)
        full_save_file_name = get_valid_write_path(full_save_file_name, extensions=[".safetensors"])
        msmodelslim_logger.info(f"Start save {full_save_file_name}")
        with SafeWriteUmask(umask=0o377):
            save_file(self.wait_save_keys, full_save_file_name)
        self.saved_keys_map.update({key: save_file_name for key in self.wait_save_keys.keys()})
        self.wait_save_keys.clear()
        self._wait_save_size = 0
        msmodelslim_logger.info(f"End save {full_save_file_name}")
        
        

