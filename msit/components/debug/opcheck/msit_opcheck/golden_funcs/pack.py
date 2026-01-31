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
from msit_opcheck.conversion.shape_convert import update_axis_for_npu_inner_format, is_transformable


class PackOperation(OperationTest):
    def golden_calc(self, in_tensors):
        ori_shape, ori_format = None, None
        input_format = self.op_param['input_desc'][0]['layout']
        for attr in self.op_param['attr']:
            if attr['key'] == 'axis':
                axis = attr['value']['i']
        for attr in self.op_param['input_desc'][0]['attr']:
            if attr['key'] == 'origin_format':
                ori_format = attr['value']['s']
            if attr['key'] == 'origin_shape':
                ori_shape = attr['value']['list']['i']
        if ori_shape is not None and not is_transformable(input_format, ori_format):
            axis = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
        return [np.concatenate(in_tensors, axis=axis)]

    def test_pack(self):
        self.execute()