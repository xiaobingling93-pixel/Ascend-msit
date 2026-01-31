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


class GatherOperation(OperationTest):
    def golden_calc(self, in_tensors):
        params_data = in_tensors[0]
        indices_data = in_tensors[1]
        axis = 0
        batch_dims = 0
        if len(in_tensors) > 2:
            tmp_axis = in_tensors[2]
            if tmp_axis.ndim == 0 or (tmp_axis.ndim == 1 and tmp_axis.size == 1):
                axis = tmp_axis[0]
        else:
            for attr in self.op_param['attr']:
                if attr['key'] == 'axis':
                    axis = attr['value']['i']
                if attr['key'] == 'batch_dims':
                    batch_dims = attr['value']['i']

        params_shape_len = len(params_data.shape)
        indices_shape_len = len(indices_data.shape)
        batch_dims = batch_dims if batch_dims >= 0 else batch_dims + indices_shape_len
        axis = axis if axis >= 0 else axis + params_shape_len

        tf.compat.v1.disable_eager_execution()
        params_shape = params_data.shape
        indices_shape = indices_data.shape

        params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
        indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)
        with tf.compat.v1.Session() as sess:
            gather_res = tf.compat.v1.gather(params, indices, axis=axis, batch_dims=batch_dims, name=None)
            res = sess.run(gather_res, feed_dict={params: params_data, indices: indices_data})
        return [res]

    def test_gather(self):
        self.execute()