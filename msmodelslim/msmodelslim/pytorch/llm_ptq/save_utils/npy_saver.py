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
import fnmatch

import torch
import numpy as np

from ascend_utils.common.security import get_valid_write_path, SafeWriteUmask
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantConfig, QuantType
from .tensor_collector import BaseSaver

class NpySaver(BaseSaver):
    def __init__(self, cfg: QuantConfig, save_directory: str = '.'):
        super().__init__()
        self.cfg = cfg
        self.save_directory: str = save_directory
        
        self.quant_weight_dict = {}
        self.scale_dict = {}
        self.offset_dict = {}
        self.anti_norm_wb = {}
        if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
            self.deq_scale_dict = {}
            self.quant_bias_dict = {}
        if self.use_kvcache_quant:
            self.kv_cache_scale = {}
            self.kv_cache_offset = {}
        if self.cfg.use_fa_quant:
            self.fa_quant_scale = {}
            self.fa_quant_offset = {}

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not self.is_enabled:
            return
        if key.endswith('.weight') or key.endswith('.bias'):
            self.quant_weight_dict[key] = value

    def post_process(self) -> None:
        for name, item in self.quant_param_collector.items():
            if name.endswith('.quant_bias'):
                self.quant_bias_dict[name] = item
            elif name.endswith('.deq_scale'):
                self.deq_scale_dict[name] = item
            elif name.endswith('.input_scale') or name.endswith('.weight_scale'):
                self.scale_dict[name] = item
            elif name.endswith('.input_offset') or name.endswith('.weight_offset'):
                self.offset_dict[name] = item
            elif name.endswith('.kv_cache_scale'):
                self.kv_cache_scale[name] = item
            elif name.endswith('.kv_cache_offset'):
                self.kv_cache_offset[name] = item
            elif fnmatch.fnmatch(name, '*.fa_*scale*'):
                self.fa_quant_scale[name] = item
            elif fnmatch.fnmatch(name, '*.fa_*offset*'):
                self.fa_quant_offset[name] = item
            elif fnmatch.fnmatch(name, '*norm.weight') or fnmatch.fnmatch(name, '*norm.bias'):
                self.anti_norm_wb[name] = item


        self.save_param(self.save_directory, "quant_weight.npy", self.quant_weight_dict)
        if self.cfg.model_quant_type in [QuantType.W8A8, QuantType.W8A8S]:
            self.save_param(self.save_directory, "input_scale.npy", self.scale_dict)
            self.save_param(self.save_directory, "input_offset.npy", self.offset_dict)
            self.save_param(self.save_directory, "quant_bias.npy", self.quant_bias_dict)
            self.save_param(self.save_directory, "deq_scale.npy", self.deq_scale_dict)
        if self.cfg.model_quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC]:
            self.save_param(self.save_directory, "weight_scale.npy", self.scale_dict)
            self.save_param(self.save_directory, "weight_offset.npy", self.offset_dict)
        if self.use_kvcache_quant:
            self.save_param(self.save_directory, "kv_cache_scale.npy", self.kv_cache_scale)
            self.save_param(self.save_directory, "kv_cache_offset.npy", self.kv_cache_offset)
        if self.cfg.use_fa_quant:
            self.save_param(self.save_directory, "fa_quant_scale.npy", self.fa_quant_scale)
            self.save_param(self.save_directory, "fa_quant_offset.npy", self.fa_quant_offset)
        if self.anti_norm_wb:
            self.save_param(self.save_directory, "anti_fp_norm.npy", self.anti_norm_wb)

        self.logger.info("Numpy weight saved successfully!")
        
    def save_param(self, output_path, output_name, output_file):
        output_path = os.path.join(output_path, output_name)
        self.logger.debug("The directory path for the quant param is %s ", output_path)
        output_path = get_valid_write_path(output_path)
        with SafeWriteUmask(umask=0o377):
            np.save(output_path, output_file)

