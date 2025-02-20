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

from ascend_utils.common.security import get_valid_write_path
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer.base import BaseWriter


class JsonDescriptionWriter(BaseWriter):

    def __init__(self, logger, model_quant_type, json_name=None, save_directory: str = '.',
                 use_kvcache_quant=False, use_fa_quant=False):
        super().__init__(logger)

        self.save_dir = save_directory
        self.json_name = json_name
        self.quant_model_json_description = QuantModelJsonDescription(
            model_quant_type,
            use_kvcache_quant,
            use_fa_quant
        )

    def _write(self, key: str, value: str) -> None:
        self.quant_model_json_description.change_weight_type(key, value)

    def _close(self) -> None:
        quant_model_description_path = os.path.join(self.save_dir, self.json_name)
        quant_model_description_path = get_valid_write_path(quant_model_description_path, extensions=[".json"])
        self.quant_model_json_description.save(quant_model_description_path)
