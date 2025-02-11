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

import torch

from ascend_utils.common.security import get_valid_write_path
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription, QuantConfig

from .tensor_collector import BaseSaver


class JsonDescriptionSaver(BaseSaver):

    def __init__(self, cfg: QuantConfig, save_directory: str = '.', save_prefix: str = "quant_model_description"):
        super().__init__()
        self.cfg = cfg
        self.model_quant_type = self.cfg.model_quant_type
        self.save_prefix: str = f"{save_prefix}_{self.model_quant_type}"
        self.quant_model_json_description = QuantModelJsonDescription(
            self.cfg.model_quant_type,
            self.cfg.use_kvcache_quant,
            self.cfg.use_fa_quant
        )

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not self.is_enabled:
            return
        self.quant_model_json_description.change_weight_type(key, self.model_quant_type)

    def post_process(self) -> None:
        if not self.is_enabled:
            return
        quant_model_description_path = os.path.join(self.save_path, self.json_name)
        quant_model_description_path = get_valid_write_path(quant_model_description_path, extensions=[".json"])
        self.quant_model_json_description.save(quant_model_description_path)