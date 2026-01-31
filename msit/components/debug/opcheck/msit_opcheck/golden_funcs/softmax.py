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
import copy
import numpy as np
import tensorflow as tf

from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.shape_convert import fhd2nd, nz2nd, shd2nd, nd2fhd, nd2nz, to_ndc1hwc0
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP


class SoftmaxOperation(OperationTest):
    def golden_calc(self, in_tensors):
        data_in = in_tensors[0]
        for attr in self.op_param['attr']:
            if attr['key'] == 'axes':
                axis = attr['value']['list']['i']
        for attr in self.op_param['input_desc'][0]['attr']:
            if attr['key'] == 'origin_shape':
                ori_shape = attr['value']['list']['i']
        ipt_dtype = DATA_TYPE_MAP[self.op_param['input_desc'][0]['dtype']]
        out_dtype = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        inputs = [ori_shape, data_in, axis, ipt_dtype, out_dtype]
        res = self._softmax_v2(inputs)
        return [res]

    def test_softmax(self):
        self.execute()

    def softmax(self, x, axis=None):
        is_bf16 = False
        if x.dtype == tf.bfloat16.as_numpy_dtype:
            x = x.astype("float32")
            is_bf16 = True
        reduce_max = np.amax(x, axis=axis, keepdims=True)
        sub_0 = np.subtract(x, reduce_max)
        has_improve_precision = False
        if sub_0.dtype == "float16":
            sub_0 = sub_0.astype("float32")
            has_improve_precision = True
        exp_0 = np.exp(sub_0)
        reduce_sum = np.sum(exp_0, axis=axis, keepdims=True)
        out = np.divide(exp_0, reduce_sum)
        if out.dtype == "float32" and has_improve_precision:
            out = out.astype("float16")
        if is_bf16:
            out = out.astype(tf.bfloat16.as_numpy_dtype)
        return out

    def normalize_axis(self, axis, shape_length):
        normalized_axis = []
        if isinstance(axis, int):
            normalized_axis = [axis]
        elif isinstance(axis, tuple):
            normalized_axis = list(axis)
        elif isinstance(axis, list):
            normalized_axis = copy.deepcopy(axis)
        if not normalized_axis:
            normalized_axis = [-1]
        normalized_axis = [v if v >= 0 else v + shape_length for v in normalized_axis]
        normalized_axis = tuple(list(set(normalized_axis)))
        return normalized_axis

    def _softmax_v2(self, inputs):
        
        ori_shape, data, axis, ipt_dtype, out_dtype = inputs
        if ipt_dtype == "float16":
            data = data.astype("float32")
        # the axis is corresponding to the original shape
        # normalize axis
        axis = self.normalize_axis(axis, len(ori_shape))

        # calc softmax
        result = self.softmax(data, axis)

        if out_dtype == "bfloat16":
            result = result.astype(tf.bfloat16.as_numpy_dtype, copy=False)
        else:
            result = result.astype(out_dtype, copy=False)
        return result
