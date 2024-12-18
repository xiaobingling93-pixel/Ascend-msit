# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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


def _fused_mul_add(context: OpInfo):
    input0, input1, input2 = context.param.get("input_arrays")
    tf.compat.v1.disable_eager_execution()
    input0_holder = tf.compat.v1.placeholder(shape=input0.shape,
                                      dtype=input0.dtype)
    input1_holder = tf.compat.v1.placeholder(shape=input1.shape,
                                      dtype=input1.dtype)
    input2_holder = tf.compat.v1.placeholder(shape=input2.shape,
                                      dtype=input2.dtype)
    output_data1 = tf.multiply(input0_holder, input1_holder, name="bert/mul")
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(output_data1, feed_dict={input0_holder: input0, input1_holder: input1})
    res_mul = tf.constant(res, dtype=res.dtype)
    output_data = tf.add(res_mul, input2_holder, name="fusedmuladd")
    with tf.compat.v1.Session() as sess:
        res = sess.run(output_data, feed_dict={input2_holder: input2})

    return res
