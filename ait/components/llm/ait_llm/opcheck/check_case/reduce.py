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


class OpcheckReduceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        op_type = self.op_param.get('reduceType', None)
        axis = self.op_param.get('axis', None)
        if op_type == 1:
            return [in_tensors[0].amax(axis)[0]]
        else:
            return [in_tensors[0].amin(axis)[0]]

    def test(self):
        ret = self.validate_param("reduceType", "axis")
        if not ret:
            return
        self.execute()