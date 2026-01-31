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


class ClipByValueOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input_t = in_tensors[0]
        clip_value_min = in_tensors[1]
        clip_value_max = in_tensors[2]
        if BFLOAT16 in str(input_t.dtype):
            input_t = input_t.astype(FLOAT32)
            clip_value_min = clip_value_min.astype(FLOAT32)
            clip_value_max = clip_value_max.astype(FLOAT32)
        min_ = np.minimum(input_t, clip_value_max)
        res = np.maximum(min_, clip_value_min)
        if BFLOAT16 in str(input_t.dtype):
            return [res.astype(input_t.dtype)]
        return [res]
    
    def test_clip_by_value(self):
        self.execute()