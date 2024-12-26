# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

from msit_opcheck.graph_parser import OpInfo


def _conv2d(context: OpInfo):

    tf.compat.v1.disable_eager_execution()
    x, conv_filter, bias, _ = context.param.get("input_arrays")
    _, conv_filter_ori_shape, _, _ = context.param.get("dyn_ori_inputs")
    strides = context.param.get("strides")
    pads = context.param.get("pads")
    dilations = context.param.get("dilations")
    groups = context.param.get('groups')  # ('groups', 1)
    data_format = context.param.get("data_format")  # ("data_format", "NCHW")
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    n, c, h, w, c0 = x.shape
    c1khkw, target_c1, target_c0, c0 = conv_filter.shape
    _, _, ori_h, ori_w = conv_filter_ori_shape
    target_n = n
    target_c = target_c1 * target_c0
    target_h = (h + pad_top + pad_bottom - (dilationh * (ori_h - 1) + 1)) // strideh + 1
    target_w = (w + pad_left + pad_right - (dilationw * (ori_w - 1) + 1)) // stridew + 1
    # 5HD to HWCN
    x = x.transpose(0, 2, 3, 1, 4).reshape(n, h, w, c * c0).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    # x filter to NHWC
    conv_filter = conv_filter.transpose(0, 3, 1, 2).reshape(c0 * c1khkw, target_c1, target_c0)
    conv_filter = conv_filter.reshape(c0 * c, ori_h * ori_w, target_c1, target_c0)
    conv_filter = conv_filter.transpose(1, 0, 2, 3).reshape(ori_h, ori_w, c0 * c, target_c1, target_c0)
    conv_filter = conv_filter.reshape(ori_h, ori_w, c0 * c, target_c1 * target_c0).astype(np.float32)
    if groups > 1:
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
    else:
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
    # NHWC to NC1HWC0
    res = out.reshape((target_n, target_h, target_w, target_c // c0, c0)).transpose(0, 3, 1, 2, 4).astype(np.float16)
    return res


def _conv2d_add(context: OpInfo):

    tf.compat.v1.disable_eager_execution()
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.param.get("strides")
    pads = context.param.get("pads")
    dilations = context.param.get("dilations")
    groups = context.param.get('groups')
    data_format = context.param.get("data_format")
    offset_x = context.param.get("offset_x")
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    n, c, h, w, c0 = x.shape
    wc, wh, ww, wn, _ = conv_filter.shape
    target_n = n
    target_c = wn
    target_h = (h + pad_top + pad_bottom - (dilationh * (wh - 1) + 1)) // strideh + 1
    target_w = (w + pad_left + pad_right - (dilationw * (ww - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(n, h, w, c * c0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(wh, ww, wc * c0, wn).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias

    tf_conv2d_result1 = tf.add(tf_conv2d_result, tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result1, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "add" or fusion_mode == "conv_add"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    res = out.reshape((target_n, target_h, target_w, target_c // c0, c0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return res


def _conv2d_mul(context: OpInfo):

    tf.compat.v1.disable_eager_execution()
    x, conv_filter, bias, offset_w = context.param.get("input_arrays")
    strides = context.param.get("strides")
    pads = context.param.get("pads")
    dilations = context.param.get("dilations")
    groups = context.param.get('groups')  # ('groups', 1)
    data_format = context.param.get("data_format")  # ("data_format", "NCHW")
    offset_x = context.param.get("offset_x")  # ("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape infe
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    n, c, h, w, c0 = x.shape
    wc, wh, ww, wn, _ = conv_filter.shape
    target_n = n
    target_c = wn
    target_h = (h + pad_top + pad_bottom - (dilationh * (wh - 1) + 1)) // strideh + 1
    target_w = (w + pad_left + pad_right - (dilationw * (ww - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(n, h, w, c * c0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(wh, ww, wc * c0, wn).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.multiply(tf_conv2d_result, tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "mul" or fusion_mode == "conv_mul"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    res = out.reshape((target_n, target_h, target_w, target_c // c0, c0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return res


def _conv2d_relu(context: OpInfo):
    
    tf.compat.v1.disable_eager_execution()
    x, conv_filter, bias, offset_w = context.param.get("input_arrays")
    strides = context.param.get("strides")
    pads = context.param.get("pads")
    dilations = context.param.get("dilations")
    groups = context.param.get('groups') 
    data_format = context.param.get("data_format")
    offset_x = context.param.get("offset_x")
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    n, c, h, w, c0 = x.shape
    wc, wh, ww, wn, _ = conv_filter.shape
    target_n = n
    target_c = wn
    target_h = (h + pad_top + pad_bottom - (dilationh * (wh - 1) + 1)) // strideh + 1
    target_w = (w + pad_left + pad_right - (dilationw * (ww - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(n, h, w, c * c0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(wh, ww, wc * c0, wn).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.nn.relu(tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "relu" or fusion_mode == "conv_relu"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    res = out.reshape((target_n, target_h, target_w, target_c // c0, c0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return res