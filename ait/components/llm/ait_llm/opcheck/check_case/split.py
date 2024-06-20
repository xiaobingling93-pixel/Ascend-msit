# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import torch
import torch_npu

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckAddOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        split_num = self.op_param.get('splitNum', None)
        split_dim = self.op_param.get('splitDim', None)
        split_output = torch.chunk(in_tensors[0], chunks=split_num, dim=split_dim)
        return split_output

    def test(self):
        ret = self.validate_param("splitNum", "splitDim")
        if not ret:
            return
        self.execute()