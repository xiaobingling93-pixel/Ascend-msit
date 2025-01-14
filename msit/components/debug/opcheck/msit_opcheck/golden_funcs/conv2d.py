# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

import numpy as np
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.shape_convert import format_transformation_map


class Conv2dOperation(OperationTest):
    @staticmethod
    def conv2d(x, conv_filter, bias, conv_params):
        pad_top, pad_bottom, pad_left, pad_right = conv_params['pads']
        strideh = conv_params['strideh']
        stridew = conv_params['stridew']
        dilations = conv_params['dilations']

        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
        tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                        padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                        data_format="NHWC", dilations=dilations)
        feed_dict = {tensor_x: x, tensor_filter: conv_filter}
        if bias is not None:
            bias = bias.astype(np.float32)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
            feed_dict[tensor_bias] = bias
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
        fusion_mode = None
        if fusion_mode is not None and (fusion_mode == "relu" or fusion_mode == "conv_relu"):
            out = np.maximum(out, 0)
        return out

    @staticmethod
    def conv2d_with_groups(x, conv_filter, groups, bias, conv_params):
        pad_top, pad_bottom, pad_left, pad_right = conv_params['pads']
        strideh = conv_params['strideh']
        stridew = conv_params['stridew']
        dilations = conv_params['dilations']

        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        output_c_per_group = conv_filter.shape[3] // groups
        input_c_per_group = conv_filter.shape[2] // groups
        split_data = tf.split(value=tensor_x,
                              num_or_size_splits=groups,
                              axis=3,
                              name='split1')
        if bias is not None:
            bias = bias.astype(np.float32)
        split_per_group_conv = []
        split_group_index = 0
        for data in split_data:
            conv_filter_per_group = \
                conv_filter[:, :, 0:input_c_per_group,
                split_group_index * output_c_per_group: (split_group_index + 1) * output_c_per_group]

            tf_conv2d_result = tf.nn.conv2d(data, conv_filter_per_group,
                                            strides=(strideh, stridew),
                                            padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                            data_format="NHWC",
                                            use_cudnn_on_gpu=False,
                                            dilations=dilations)
            if bias is not None:
                bias_split_data = \
                    bias[split_group_index * output_c_per_group:(split_group_index + 1) * output_c_per_group]
                tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, bias_split_data)
            split_group_index += 1
            split_per_group_conv.append(tf_conv2d_result)

        output = tf.concat(split_per_group_conv, axis=3, name='concat1')
        init_op = tf.compat.v1.global_variables_initializer()
        feed_dict = {tensor_x: x}
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(output, feed_dict=feed_dict)
        return out

    def get_conv_params(self):
        for attr in self.op_param['attr']:
            if attr['key'] == 'strides':
                strides = attr['value']['list']['i']
            if attr['key'] == 'pads':
                pads = attr['value']['list']['i']
            if attr['key'] == 'dilations':
                dilations = attr['value']['list']['i']
            if attr['key'] == 'groups':
                groups = attr['value']['i']
            if attr['key'] == 'data_format':
                data_format = attr['value']['s']
        # Collect shape info
        conv_params = {}
        conv_params['dilations'] = dilations
        conv_params['pads'] = pads
        h_index = data_format.index("H")
        w_index = data_format.index("W")
        conv_params['strideh'] = strides[h_index]
        conv_params['stridew'] = strides[w_index]
        return conv_params, groups

    def golden_calc(self, in_tensors):
        tf.compat.v1.disable_eager_execution()
        x = in_tensors[0]
        conv_filter = in_tensors[1]
        bias = in_tensors[2] if len(in_tensors) > 2 else None
        # get input params
        for attr in self.op_param['input_desc'][0]['attr']:
            if attr['key'] == 'origin_format':
                x_ori_format = attr['value']['s']
            if attr['key'] == 'origin_shape':
                x_ori_shape = attr['value']['list']['i']
        for attr in self.op_param['input_desc'][1]['attr']:
            if attr['key'] == 'origin_format':
                conv_filter_ori_format = attr['value']['s']
            if attr['key'] == 'origin_shape':
                conv_filter_ori_shape = attr['value']['list']['i']
        conv_filter_new_format = self.op_param['input_desc'][1]['layout']
        x_new_format = self.op_param['input_desc'][0]['layout']
        conv_params, groups = self.get_conv_params()
        x = x.astype(np.float32)
        conv_filter = conv_filter.astype(np.float32)

        # 5HD to HWCN
        if len(x.shape) != 5:
            raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
        x = format_transformation_map[x_new_format][x_ori_format](x, x_new_format, x_ori_shape)
        # x filter to NHWC
        if conv_filter_ori_format != conv_filter_new_format:
            conv_filter = format_transformation_map[conv_filter_new_format][conv_filter_ori_format]\
                (conv_filter, conv_filter_new_format, conv_filter_ori_shape)
        if groups > 1:
            out = self.conv2d_with_groups(x, conv_filter, groups, bias, conv_params)
        else:
            out = self.conv2d(x, conv_filter, bias, conv_params)
        for attr in self.op_param['output_desc'][0]['attr']:
            if attr['key'] == 'origin_format':
                out_ori_format = attr['value']['s']
        out_format = self.op_param['output_desc'][0]['layout']
        out_target_shape = self.op_param['output_desc'][0]['shape']['dim']
        # output shape convert
        if out_format != out_ori_format:
            out = format_transformation_map[out_ori_format][out_format](out, out_ori_format, out_target_shape)
        return [out]

    def test_conv2d(self):
        self.execute()
