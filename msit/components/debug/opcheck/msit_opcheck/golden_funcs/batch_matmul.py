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
from msit_opcheck.conversion.dtype_convert import bfloat16_conversion_v2, DATA_TYPE_MAP
from msit_opcheck.operation_test import OperationTest
from msit_opcheck.golden_funcs.mat_mul import matmul


class BatchMatMulOperation(OperationTest):
    def golden_calc(self, in_tensors):
        # input & params
        # 校验输入张量列表的长度，至少需要2个输入
        if len(in_tensors) < 2:
            raise ValueError("Insufficient input tensors, at least 2 input tensors are required")
        x1 = in_tensors[0]
        x2 = in_tensors[1]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        for attr in self.op_param['attr']:
            if attr['key'] == 'adj_x1':
                trans_a = attr['value']['b']
            if attr['key'] == 'adj_x2':
                trans_b = attr['value']['b']
        # bias
        bias = None
        if len(in_tensors) > 2:
            bias = in_tensors[2]

        inputs = [x1, x2, trans_a, trans_b, out_dtype, bias]
        res = matmul(inputs)
        return [res]

    def test_batch_matmul(self):
        self.execute()
