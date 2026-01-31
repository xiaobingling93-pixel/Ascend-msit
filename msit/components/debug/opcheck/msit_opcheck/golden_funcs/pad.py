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
from collections import deque
import numpy as np
import torch

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.constants import BFLOAT16


class PadOperation(OperationTest):
    def golden_calc(self, in_tensors):
        input_data, paddings = in_tensors
        for attr in self.op_param['input_desc'][0]['attr']:
            if attr['key'] == 'origin_format':
                input_format = attr['value']['s']
        bf16_mark = False
        if True:
            pad_shape = deque()
            for i in range(len(paddings)):
                pad_shape.append(paddings[len(paddings) - 1 - i][0])
                pad_shape.append(paddings[len(paddings) - 1 - i][1])

            if (input_format == "NC1HWC0"):
                pad_shape.appendleft(0)
                pad_shape.appendleft(0)
            if BFLOAT16 in str(input_data.dtype):
                bf16_mark = True
                input_data = input_data.astype(np.float32)
            input_data_tensor = torch.from_numpy(input_data)
            golden = torch.constant_pad_nd(input_data_tensor, tuple(pad_shape), 0)
            if bf16_mark:
                golden.to(torch.bfloat16)
            res = golden.numpy()
        return [res]

    def test_pad(self):
        self.execute()


