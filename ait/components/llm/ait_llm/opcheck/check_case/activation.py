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
import torch.nn.functional as F

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class ActivationGolden:
    @staticmethod
    def relu_golden(in_tensors, _):
        return torch.nn.functional.relu(in_tensors)

    @staticmethod
    def gelu_golden(in_tensors, _):
        in_tensors = in_tensors.float()
        try:
            float_result = 0.5 * in_tensors * (1 + torch.nn.functional.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * 
                                                    (in_tensors + 0.044715 * torch.pow(in_tensors, 3))))
        except ZeroDivisionError as e:
            raise e
        return float_result.half() if in_tensors.dtype == torch.float16 else float_result

    @staticmethod
    def fast_gelu_golden(in_tensors, _):
        in_tensors = in_tensors.float()
        try:
            float_result = in_tensors * torch.exp(0.851 * (in_tensors - torch.abs(in_tensors))) / (1 + 
                            torch.exp(-1.702 * torch.abs(in_tensors)))
        except ZeroDivisionError as e:
            raise e
        return float_result.half()

    @staticmethod
    def swish_golden(in_tensors, scale):
        in_tensors = in_tensors.float()
        try:
            float_result = in_tensors / (1 + torch.exp(-in_tensors * scale))
        except ZeroDivisionError as e:
            raise e
        return float_result.half()

    @staticmethod
    def log_golden(in_tensors, _):
        in_tensors = in_tensors.float()
        float_result = torch.log(in_tensors)
        return float_result.half() if in_tensors.dtype == torch.float16 else float_result
    
    @staticmethod
    def swigluforward_golden(in_tensors, dim):
        dtype = in_tensors.dtype
        float_in_tensors = in_tensors.float()
        a, b = float_in_tensors.chunk(2, dim)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        float_result = F.silu(a) * b
        return float_result.to(dtype)
    
    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)
    
    @staticmethod
    def swish_grad(x):
        return torch.sigmoid(x) + x * (1 - torch.sigmoid(x)) * torch.sigmoid(x)
    
    @staticmethod
    def swiglubackward_golden(in_tensors, dim):
        dtype = in_tensors[1].dtype
        tensor_y_grad = in_tensors[1].float()
        x = in_tensors[1].float()
        a, b = x.chunk(2, dim)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        y1 = b * tensor_y_grad * ActivationGolden.swish_grad(a)
        y2 = tensor_y_grad * ActivationGolden.swish(a)
        y = torch.concat((y1, y2), dim)
        return y.to(dtype)        


class OpcheckActivationOperation(operation_test.OperationTest):
    golden_func = {
        1: ActivationGolden.relu_golden,
        2: ActivationGolden.gelu_golden,
        3: ActivationGolden.fast_gelu_golden,
        4: ActivationGolden.swish_golden,
        5: ActivationGolden.log_golden,
        6: ActivationGolden.swigluforward_golden,
        7: ActivationGolden.swiglubackward_golden,
    } 

    def golden_calc(self, in_tensors):
        activation_type = self.op_param.get("activationType", None)
        scale = self.op_param.get("scale", None)
        dim = self.op_param.get("dim", None)
        if activation_type == 6:
            golden_result = OpcheckActivationOperation.golden_func[activation_type](in_tensors[0], dim)
        elif activation_type == 7:
            golden_result = OpcheckActivationOperation.golden_func[activation_type](in_tensors, dim)
        else:
            golden_result = OpcheckActivationOperation.golden_func[activation_type](in_tensors[0], scale)
        return [golden_result]

    def test(self):
        ret = self.validate_param("activationType")
        if not ret:
            return
        self.execute()