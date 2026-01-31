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
import numpy as np

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.constants import FLOAT32, BFLOAT16


class ReluOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input0 = in_tensors[0]
        if BFLOAT16 in str(input0.dtype):
            x = input0.astype(FLOAT32)
            res = np.maximum(x, 0)
            res = res.astype(input0.dtype, copy=False)
        else:
            res = np.maximum(input0, 0)
        return [res]

    def test_relu(self):
        self.execute()