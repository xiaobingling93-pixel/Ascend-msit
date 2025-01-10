# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import numpy as np
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.shape_convert import transform
from msit_opcheck.conversion.shape_convert import format_transformation_map


class TileDOperation(OperationTest):
    def golden_calc(self, in_tensors):
        tf.compat.v1.disable_eager_execution()
        x = in_tensors[0]
        for attr in self.op_param['attr']:
            if attr['key'] == 'multiples':
                multiples = attr['value']['list']['i']

        # input format转换
        for attr in self.op_param['input_desc'][0]['attr']:
            if attr['key'] == 'origin_format':
                x_ori_format = attr['value']['s']
            if attr['key'] == 'origin_shape':
                x_ori_shape = attr['value']['list']['i']
        x_new_format = self.op_param['input_desc'][0]['layout']
        if x_new_format != x_ori_format:
            x = format_transformation_map[x_new_format][x_ori_format](x, x_new_format, x_ori_shape)

        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        out = tf.tile(tensor_x, multiples)
        with tf.compat.v1.Session() as sess:
            res = sess.run(out, feed_dict={tensor_x: x})

        for attr in self.op_param['output_desc'][0]['attr']:
            if attr['key'] == 'origin_format':
                output_ori_format = attr['value']['s']
        output_format = self.op_param['output_desc'][0]['layout']
        output_shape = self.op_param['output_desc'][0]['shape']['dim']
        if transform(res, output_ori_format, output_format, output_shape) is not None:
            res = transform(res, output_ori_format, output_format, output_shape)
        return [res]

    def test_tiled(self):
        self.execute()