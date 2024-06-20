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


class OpcheckLayerNormOperation(operation_test.OperationTest):
    def layer_norm_quant(self, layer_norm_res, quant_scale, quant_offset):
        golden_result_quant = layer_norm_res * quant_scale + quant_offset
        golden_result_quant = torch.round(golden_result_quant)
        golden_result_quant = torch.clamp(golden_result_quant, -128, 127)
        return golden_result_quant

    def golden_calc(self, in_tensors):
        layer_type = self.op_param.get('layerType', None)
        if layer_type == 1:
            cur_param = self.op_param.get('normParam', None)
        elif layer_type == 3:
            cur_param = self.op_param.get('postNormParam', None)
        else:
            raise ValueError('layerType should be 1 or 3')
        
        eps = cur_param.get('epsilon', 1e-5)
        is_quant = cur_param.get('quantType', 0)
        quant_scale = cur_param.get('quantInputScale', 1)
        quant_offset = cur_param.get('quantInputOffset', 0)
        quant_alpha = cur_param.get('quantInputAlpha', 1)

        if not is_quant:
            if layer_type == 1:
                op_input = in_tensors[0]
                weight = in_tensors[1]
                bias = in_tensors[2]
                axis = cur_param.get('beginNormAxis', 0)
                normalized_shape = in_tensors[0].shape[axis:]
                golden_result = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
            elif layer_type == 3:
                weight = in_tensors[2]
                bias = in_tensors[3]
                normalized_shape = (1, in_tensors[0].shape[-1])
                zoom_scale_value = cur_param.get('zoomScale', 1)
                op_input = torch.add(in_tensors[0], zoom_scale_value * in_tensors[1])
                golden_result = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
            golden = [golden_result.half()] if in_tensors[0].dtype == torch.float16 else [golden_result]
        else:
            if layer_type == 1:
                op_input = in_tensors[0]
                weight = in_tensors[1]
                bias = in_tensors[2]                    
                normalized_shape = (1, in_tensors[0].shape[-1])
                layer_norm_res = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
                golden_result = layer_norm_res * quant_alpha
                golden_result_quant = self.layer_norm_quant(layer_norm_res, quant_scale, quant_offset)
            elif layer_type == 3:
                weight = in_tensors[2]
                bias = in_tensors[3]
                normalized_shape = (1, in_tensors[0].shape[-1])                
                op_input = torch.add(in_tensors[0], in_tensors[1])
                layer_norm_res = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
                golden_result = (layer_norm_res * quant_alpha)
                golden_result_quant = self.layer_norm_quant(layer_norm_res, quant_scale, quant_offset)
            golden = [golden_result, golden_result_quant]        

        return golden

    def test(self):
        ret = self.validate_param("layerType")
        if not ret:
            return
        self.execute()