# -*- coding: utf-8 -*-
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
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP, numpy_to_torch_tensor
from msit_opcheck.constants import FLOAT32, FLOAT16, BFLOAT16


class TanhOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input_x = in_tensors[0]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        if input_x.dtype == FLOAT16:
            input_x = input_x.astype(FLOAT32, copy=False)
            input_x = numpy_to_torch_tensor(input_x)
        elif BFLOAT16 in str(input_x.dtype):
            np_fp32 = input_x.astype(FLOAT32)
            t_fp32 = torch.from_numpy(np_fp32)
            input_x = t_fp32.to(torch.bfloat16, copy=False)
        else:
            input_x = numpy_to_torch_tensor(input_x)
        res = torch.tanh(input_x)

        if str(out_dtype) == BFLOAT16:
            res = res.to(torch.float32).numpy().astype(tf.bfloat16.as_numpy_dtype)
        else:
            res = res.numpy().astype(out_dtype)

        return [res]

    def test_tanh(self):
        self.execute()