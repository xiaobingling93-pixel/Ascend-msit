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
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP
from msit_opcheck.utils import broadcast_to_maxshape
from msit_opcheck.constants import BFLOAT16, FLOAT16


class SelectOperation(OperationTest):
    def golden_calc(self, in_tensors):
        condition, x1, x2 = in_tensors[:3]
        output_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        shape_x1 = x1.shape
        shape_x2 = x2.shape
        shape_condition = condition.shape
        shape_list = broadcast_to_maxshape([shape_x1, shape_x2, shape_condition])
        x1 = np.broadcast_to(x1, shape_list[-1])
        x2 = np.broadcast_to(x2, shape_list[-1])
        condition = np.broadcast_to(condition, shape_list[-1])
        ones = np.ones(shape_list[-1], dtype=FLOAT16)
        equal_var = np.equal(condition, ones)
        if output_dtype == BFLOAT16:
            return [np.where(equal_var, x1, x2).astype(tf.bfloat16.as_numpy_dtype)]
        else:
            return [np.where(equal_var, x1, x2).astype(output_dtype)]

    def test_select(self):
        self.execute()