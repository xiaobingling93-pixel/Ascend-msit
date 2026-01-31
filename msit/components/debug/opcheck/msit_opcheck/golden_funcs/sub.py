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
import torch
import tensorflow.compat.v1 as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.utils import broadcast_to_maxshape


class SubOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x1 = in_tensors[0]
        x2 = in_tensors[1]
        type_ = x1.dtype
        if type_ == "bool":
            x1, x2 = torch.tensor(x1), torch.tensor(x2)
            res = torch.logical_xor(x1, x2).numpy()
        else:
            shape_list = broadcast_to_maxshape([x1.shape, x2.shape])
            x1 = np.broadcast_to(x1, shape_list[-1])
            x2 = np.broadcast_to(x2, shape_list[-1])
            tf.disable_v2_behavior()
            tensor_x1 = tf.placeholder(x1.dtype, shape=x1.shape)
            tensor_x2 = tf.placeholder(x2.dtype, shape=x2.shape)
            out = tf.subtract(tensor_x1, tensor_x2)
            feed_dict = {tensor_x1: x1, tensor_x2: x2}
            with tf.Session() as sess:
                res = sess.run(out, feed_dict=feed_dict)
        return [res]

    def test_sub(self):
        self.execute()