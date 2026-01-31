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

from msit_opcheck.utils import broadcast_to_maxshape
from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP
from msit_opcheck.constants import FLOAT16


class LogicalOrOperation(OperationTest):
    def golden_calc(self, in_tensors):
        # 校验输入张量列表长度
        if len(in_tensors) < 2:
            raise ValueError("Insufficient input tensors, at least 2 input tensors are required")
            
        x1, x2 = in_tensors[:2]
        
        # 校验op_param字典结构
        if not isinstance(self.op_param, dict) or 'output_desc' not in self.op_param:
            raise ValueError("Invalid op_param format")
        if not isinstance(self.op_param['output_desc'], list) or len(self.op_param['output_desc']) == 0:
            raise ValueError("Invalid output_desc format")
        if not isinstance(self.op_param['output_desc'][0], dict) or 'dtype' not in self.op_param['output_desc'][0]:
            raise ValueError("Invalid output_desc[0] format")
            
        # 校验dtype是否在数据类型映射表中
        dtype = self.op_param['output_desc'][0]['dtype']
        if dtype not in DATA_TYPE_MAP:
            raise ValueError(f"Unsupported data type: {dtype}")
            
        out_type = DATA_TYPE_MAP[dtype]
        shape_list = broadcast_to_maxshape([x1.shape, x2.shape])
        x1 = x1.astype(FLOAT16)
        x2 = x2.astype(FLOAT16)
        x1 = np.broadcast_to(x1, shape_list[-1])
        x2 = np.broadcast_to(x2, shape_list[-1])
        return [np.maximum(x1, x2).astype(out_type, copy=False)]

    def test_logical_or(self):
        self.execute()
