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

import copy
from typing import Tuple

import numpy
import tensorflow as tf

from msit_opcheck.conversion import shape_convert
from msit_opcheck.graph_parser import OpInfo


def softmax(x, axis=None):
    is_bf16 = False
    if x.dtype == tf.bfloat16.as_numpy_dtype:
        x = x.astype("float32")
        is_bf16 = True
    reduce_max = numpy.amax(x, axis=axis, keepdims=True)
    sub_0 = numpy.subtract(x, reduce_max)
    has_improve_precision = False
    if sub_0.dtype == "float16":
        sub_0 = sub_0.astype("float32")
        has_improve_precision = True
    exp_0 = numpy.exp(sub_0)
    reduce_sum = numpy.sum(exp_0, axis=axis, keepdims=True)
    out = numpy.divide(exp_0, reduce_sum)
    if out.dtype == "float32" and has_improve_precision:
        out = out.astype("float16")
    if is_bf16:
        out = out.astype(tf.bfloat16.as_numpy_dtype)
    return out


def normalize_axis(axis, shape_length) -> Tuple:
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


def _softmax_v2(context: OpInfo):
    format = context.param.get("stc_input_formats")[0]
    ori_format = context.param.get("stc_input_ori_formats")[0]
    ori_shape = context.param.get("stc_ori_inputs")[0]
    data = context.param.get("input_arrays")[0]
    axis = context.param.get("axes")
    ipt_dtype = context.param.get("stc_input_dtypes")[0]
    out_dtype = context.param.get("output_dtypes")[0]

    if ipt_dtype == "float16":
        data = data.astype("float32")
    # the axis is corresponding to the original shape
    # normalize axis
    axis = normalize_axis(axis, len(ori_shape))

    # convert any format to ND
    if format == "NC1HWC0":
        data = shape_convert.fhd2nd(data, ori_shape, ori_format)
    elif format == "FRACTAL_NZ":
        data = shape_convert.nz2nd(data, ori_shape)
    elif format == "NDC1HWC0":
        data = shape_convert.shd2nd(data, ori_shape, ori_format)

    # calc softmax
    result = softmax(data, axis)

    # convert ND to target format
    if format == "NC1HWC0":
        result = shape_convert.nd2fhd(result, ori_format, context.param.get("stc_outputs")[0])
    elif format == "FRACTAL_NZ":
        result = shape_convert.nd2nz(result, context.param.get("stc_outputs")[0])
    elif format == "NDC1HWC0":
        result = shape_convert.to_NDC1HWC0(result, ori_format, context.param.get("stc_outputs")[0])
    if out_dtype == "bfloat16":
        result = result.astype(tf.bfloat16.as_numpy_dtype, copy=False)
    else:
        result = result.astype(out_dtype, copy=False)
    return result