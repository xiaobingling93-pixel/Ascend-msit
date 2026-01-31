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


class BiasAddOperation(OperationTest):
    def golden_calc(self, in_tensors):
        value, bias = in_tensors
        bias_shape = [1] * (value.ndim - 1) + [bias.shape[0]]
        bias_reshape = bias.reshape(bias_shape)
        res = np.add(value, bias_reshape)
        return [res]
    
    def test_bias_add(self):
        self.execute()

