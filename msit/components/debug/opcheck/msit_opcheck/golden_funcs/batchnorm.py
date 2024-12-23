# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import numpy as np

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP


class BatchNormOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input0_dtype = DATA_TYPE_MAP[self.op_param['input_desc'][0]['dtype']]
        input1_dtype = DATA_TYPE_MAP[self.op_param['input_desc'][1]['dtype']]
        output_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        data_x, data_weight, data_bias, data_running_mean, data_running_var = in_tensors
        momentum = 0.1
        for attr in self.op_param['attr']:
            if attr['key'] == 'epsilon':
                eps = float(attr['value']['f'])
        for attr in self.op_param['output_desc'][0]['attr']:
            if attr['key'] == "origin_format":
                data_format = attr['value']['s']

        if output_dtype == "bfloat16" or output_dtype == "float16":
            data_x = data_x.astype("float32")
            data_weight = data_weight.astype("float32")
            data_bias = data_bias.astype("float32")
        if input0_dtype == "float32" and input1_dtype == "float64":
            data_x = data_x.astype("float64")
            data_weight = data_weight.astype("float64")
            data_bias = data_bias.astype("float64")
        
        is_training = False
        
        tensor_x = torch.from_numpy(data_x).squeeze(0)  # 1NHWC--->NHWC
        tensor_weight = torch.from_numpy(data_weight).squeeze(0)
        tensor_bias = torch.from_numpy(data_bias).squeeze(0)
        tensor_running_mean = torch.from_numpy(data_running_mean).squeeze(0)
        tensor_running_var = torch.from_numpy(data_running_var).squeeze(0)

        if data_format == "NHWC":
            tensor_x = tensor_x.permute(0, 3, 1, 2)
            dims = tensor_x.shape

        res = torch.ops.aten.native_batch_norm(input=tensor_x.reshape(dims[0] * dims[1], dims[2], dims[3]), 
                                               weight=tensor_weight.view(dims[0] * dims[1]), 
                                               bias=tensor_bias.view(dims[0] * dims[1]),
                                               running_mean=tensor_running_mean.view(dims[0] * dims[1]), 
                                               running_var=tensor_running_var.view(dims[0] * dims[1]),
                                               training=is_training, momentum=momentum, eps=eps)
        
        output = res[0]
        if data_format == "NHWC":
            output = output.reshape(dims[0], dims[1], dims[2], dims[3]).permute(0, 2, 3, 1)
        return np.expand_dims(output, axis=0)

    def test_batch_norm(self):
        self.execute()
