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
import tensorflow as tf
import numpy as np

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP, numpy_to_torch_tensor
from msit_opcheck.constants import FLOAT32, BFLOAT16


class SigmoidOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input0 = in_tensors[0]
        if input0.dtype == tf.bfloat16.as_numpy_dtype:
            input0 = input0.astype(FLOAT32)
        tensor_neg = input0 * (-1)
        tensor_exp = np.exp(tensor_neg)
        tensor_add = tensor_exp + 1
        res = 1 / tensor_add
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        if out_dtype == BFLOAT16:
            res = res.astype(tf.bfloat16.as_numpy_dtype, copy=False)
        else:
            res = res.astype(out_dtype, copy=False)

        return [res]

    def test_sigmoid(self):
        self.execute()
        
