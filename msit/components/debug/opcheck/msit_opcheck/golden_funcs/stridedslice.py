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
import tensorflow.compat.v1 as tf

from msit_opcheck.operation_test import OperationTest


class StridedSliceOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x = in_tensors[0]
        if len(in_tensors) > 1:
            begin = in_tensors[1]
            end = in_tensors[2]
            strides = in_tensors[3]
        else:
            for attr in self.op_param['attr']:
                if attr['key'] == 'begin':
                    begin = attr['value']['list']['i']
                if attr['key'] == 'end':
                    end = attr['value']['list']['i']
                if attr['key'] == 'strides':
                    strides = attr['value']['list']['i']
        for attr in self.op_param['attr']:
            if attr['key'] == 'begin_mask':
                begin_mask = attr['value']['i']
            if attr['key'] == 'end_mask':
                end_mask = attr['value']['i']
            if attr['key'] == 'ellipsis_mask':
                ellipsis_mask = attr['value']['i']
            if attr['key'] == 'new_axis_mask':
                new_axis_mask = attr['value']['i']
            if attr['key'] == 'shrink_axis_mask':
                shrink_axis_mask = attr['value']['i']
        tf.disable_v2_behavior()
        x_holder = tf.placeholder(x.dtype, shape=x.shape)
        res = tf.strided_slice(x_holder, begin, end, strides, begin_mask, end_mask,
                            ellipsis_mask, new_axis_mask, shrink_axis_mask)
        with tf.Session() as sess:
            result = sess.run(res, feed_dict={x_holder: x})
        return [result]

    def test_stride_slice(self):
        self.execute()
