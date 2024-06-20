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


class OpcheckFillOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        with_mask = self.op_param.get("withMask", None)
        out_dim = self.op_param.get("out_dim", None)
        value = self.op_param.get("value", None)

        if with_mask:
            golden_result = in_tensors[0].masked_fill_(in_tensors[1], value[0])
        else:
            golden_result = torch.full(out_dim, value[0], dtype=torch.float16)
        return [golden_result]

    def test(self):
        ret = self.validate_param("withMask", "out_dim", "value")
        if not ret:
            return
        self.execute()