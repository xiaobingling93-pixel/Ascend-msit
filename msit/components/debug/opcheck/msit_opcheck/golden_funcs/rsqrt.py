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
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP
from msit_opcheck.constants import BFLOAT16


class RsqrtOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input0 = in_tensors[0]
        input0 = torch.tensor(input0, dtype=torch.float32)
        output_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        res = torch.rsqrt(input0).numpy()
        if output_dtype == BFLOAT16:
            res = res.astype(tf.bfloat16.as_numpy_dtype, copy=False)
        else:
            res = res.astype(output_dtype, copy=False)
        return [res]
    
    def test_rsqrt(self):
        self.execute()