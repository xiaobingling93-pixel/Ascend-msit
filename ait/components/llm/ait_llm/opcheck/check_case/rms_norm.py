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


class OpcheckRmsNormOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        layertype = self.op_param.get('layerType', None)
        if layertype == 1:
            cur_param = self.op_param.get('normParam', None)
        elif layertype == 2:
            cur_param = self.op_param.get('preNormParam', None)
        elif layertype == 3:
            cur_param = self.op_param.get('postNormParam', None)
        else:
            raise ValueError('layerType should be 1 or 2 or 3')

        quant_type = cur_param.get('quantType', None)
        eps = cur_param.get('epsilon', 1e-5)
        x = in_tensors[0].float()
        gamma = in_tensors[1].float()
        gamma = gamma.view(1, -1)
        if layertype == 2 and quant_type == 2:
            x = x + in_tensors[1].float()
            gamma = in_tensors[2].float()
        if layertype == 3 or (layertype == 2 and quant_type == 0):
            idx = 1
            if 'hasBias' in cur_param.keys():
                x = x + in_tensors[idx].float()
                idx += 1
            x = x + in_tensors[idx].float()
            idx += 1
            gamma = in_tensors[idx].float()
        gamma_size = float(gamma.size(-1))
        try:
            norm = torch.sum(x / gamma_size * x, dim=-1, keepdim=True) + eps
            golden_output = x * gamma / torch.sqrt(norm)
        except ZeroDivisionError as e:
            raise e
        
        def rms_norm_quant_with_tensor(golden_output, beta, scale, offset):
            golden_output = golden_output.float()
            beta = beta.float()
            scale = scale.half()
            golden_output = golden_output + beta
            try:
                golden_output = golden_output / scale + offset
            except ZeroDivisionError as e:
                raise e
            golden_output = torch.clamp(golden_output, -128, 127)
            golden_result_quant = torch.round(golden_output)
            return golden_result_quant.type(torch.int8)

        if layertype == 2 and quant_type == 2:
            golden_result = [rms_norm_quant_with_tensor(golden_output, in_tensors[3], in_tensors[4], in_tensors[5]), x]
        elif layertype == 1 and quant_type == 2:
            golden_result = [rms_norm_quant_with_tensor(golden_output, in_tensors[2], in_tensors[3], in_tensors[4])]
        elif layertype == 2 and quant_type == 0:
            golden_result = [golden_result.half(), x.half()]
        else:
            golden_result = [golden_output.half()]

        return golden_result

    def test(self):
        ret = self.validate_param("layerType")
        if not ret:
            return
        self.execute()