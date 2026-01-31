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
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP, numpy_to_torch_tensor, bfloat16_conversion


class MinimumOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x1 = in_tensors[0]
        x2 = in_tensors[1]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]

        res = np.minimum(x1, x2)
        out_dtype = bfloat16_conversion(out_dtype)
        res = res.astype(out_dtype[0])
        return [res]

    def test_minimum(self):
        self.execute()