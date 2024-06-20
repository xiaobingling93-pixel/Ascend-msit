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


class OpcheckElewiseAddOperation(operation_test.OperationTest):
    def elewise_cast(self, in_tensors):
        out_type = self.op_param.get('outTensorType', None)
        if not out_type:
            msg = "The golden data of elewise cast may be incorrect because outTensorType is not correctly set!"
            logger.error(msg)            
        golden_result = in_tensors[0]
        if out_type == 0:
            golden_result = in_tensors[0].float()
        elif out_type == 1:
            golden_result = in_tensors[0].half()
        elif out_type == 3:
            golden_result = in_tensors[0].int()
        elif out_type == 9:
            golden_result = in_tensors[0].long()
        return [golden_result]

    def elewise_muls(self, in_tensors):
        mulsParam = self.op_param.get('mulsParam', {})
        var_attr = mulsParam.get('varAttr', None)
        if not var_attr:
            msg = "The golden data of elewise muls may be incorrect because varAttr is not correctly set!"
            logger.error(msg)
            var_attr = 1
        golden_result = in_tensors[0] * var_attr
        return [golden_result]

    def elewise_cos(self, in_tensors):
        golden_result = torch.cos(in_tensors[0])
        return [golden_result.half()]

    def elewise_sin(self, in_tensors):
        golden_result = torch.sin(in_tensors[0])
        return [golden_result.half()]

    def elewise_neg(self, in_tensors):
        golden_result = in_tensors[0] * (-1.0)
        return [golden_result]

    def elewise_quant(self, in_tensors):
        golden_result = in_tensors[0].type(torch.int8)
        return [golden_result]

    def elewise_logical_not(self, in_tensors):
        golden_result = torch.logical_not(in_tensors[0])
        return [golden_result]

    def elewise_add(self, in_tensors):
        golden_result = in_tensors[0] + in_tensors[1]
        return [golden_result]

    def elewise_mul(self, in_tensors):
        golden_result = in_tensors[0] * in_tensors[1]
        return [golden_result]

    def elewise_realdiv(self, in_tensors):
        golden_result = torch.div(in_tensors[0], in_tensors[1])
        return [golden_result]

    def elewise_logical_and(self, in_tensors):
        golden_result = torch.logical_and(in_tensors[0].type(torch.bool), in_tensors[1].type(torch.bool))
        return [golden_result.type(torch.int8)]

    def elewise_logical_or(self, in_tensors):
        golden_result = torch.logical_or(in_tensors[0].type(torch.bool), in_tensors[1].type(torch.bool))
        return [golden_result.type(torch.int8)]

    def elewise_less(self, in_tensors):
        golden_result = torch.lt(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]

    def elewise_greater(self, in_tensors):
        golden_result = torch.gt(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]

    def elewise_sub(self, in_tensors):
        golden_result = in_tensors[0] - in_tensors[1]
        return [golden_result]

    def elewise_equal(self, in_tensors):
        golden_result = torch.eq(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]
        
    def golden_calc(self, in_tensors):
        elewise_type = self.op_param.get("elewiseType", None)
        if elewise_type == 1:
            golden = self.elewise_cast(in_tensors)
        elif elewise_type == 2:
            golden = self.elewise_muls(in_tensors)
        elif elewise_type == 3:
            golden = self.elewise_cos(in_tensors)
        elif elewise_type == 4:
            golden = self.elewise_sin(in_tensors)
        elif elewise_type == 5:
            golden = self.elewise_neg(in_tensors)
        elif elewise_type == 6:
            golden = self.elewise_quant(in_tensors)
        elif elewise_type == 7:
            golden = self.elewise_logical_not(in_tensors)
        elif elewise_type == 8:
            golden = self.elewise_add(in_tensors)
        elif elewise_type == 9:
            golden = self.elewise_mul(in_tensors)
        elif elewise_type == 10:
            golden = self.elewise_realdiv(in_tensors)
        elif elewise_type == 11:
            golden = self.elewise_logical_and(in_tensors)
        elif elewise_type == 12:
            golden = self.elewise_logical_or(in_tensors)
        elif elewise_type == 13:
            golden = self.elewise_less(in_tensors)
        elif elewise_type == 14:
            golden = self.elewise_greater(in_tensors)
        elif elewise_type == 15:
            golden = self.elewise_sub(in_tensors)
        elif elewise_type == 16:
            golden = self.elewise_equal(in_tensors)
        
        return golden

    def test(self):
        ret = self.validate_param("elewiseType")
        if not ret:
            return
        
        elewise_type = self.op_param.get("elewiseType", None)
        msg = f"elewiseType: {elewise_type}"
        logger.debug(msg)

        self.execute()