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
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest


class TileDOperation(OperationTest):
    def golden_calc(self, in_tensors):
        tf.compat.v1.disable_eager_execution()
        x = in_tensors[0]
        for attr in self.op_param['attr']:
            if attr['key'] == 'multiples':
                multiples = attr['value']['list']['i']
        if 'multiples' not in locals():
            multiples = in_tensors[1]

        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        out = tf.tile(tensor_x, multiples)
        with tf.compat.v1.Session() as sess:
            res = sess.run(out, feed_dict={tensor_x: x})
        return [res]

    def test_tiled(self):
        self.execute()