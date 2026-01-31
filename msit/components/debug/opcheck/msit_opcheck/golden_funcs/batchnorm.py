# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
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

        if output_dtype == "bfloat16" or output_dtype == "float16":
            data_x = data_x.astype("float32")
            data_weight = data_weight.astype("float32")
            data_bias = data_bias.astype("float32")
        if input0_dtype == "float32" and input1_dtype == "float64":
            data_x = data_x.astype("float64")
            data_weight = data_weight.astype("float64")
            data_bias = data_bias.astype("float64")
        
        is_training = False
        res = torch.ops.aten.native_batch_norm(input=torch.from_numpy(data_x),
                                               weight=torch.from_numpy(data_weight).view(-1),
                                               bias=torch.from_numpy(data_bias).view(-1),
                                               running_mean=torch.from_numpy(data_running_mean).view(-1),
                                               running_var=torch.from_numpy(data_running_var).view(-1),
                                               training=is_training, momentum=momentum, eps=eps)
        
        output = res[0]
        return [np.expand_dims(output, axis=0)]

    def test_batch_norm(self):
        self.execute()
