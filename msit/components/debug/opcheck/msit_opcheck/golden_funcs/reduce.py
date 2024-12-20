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

import functools

import numpy
import tensorflow as tf

from msit_opcheck.graph_parser import OpInfo
from msit_opcheck.conversion.dtype_convert import bfloat16_conversion

D_BOUNDS = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int16": (-32768, 32767),
    "uint16": (0, 65535),
    "int32": (-2147483648, 2147483647),
    "uint32": (0, 4294967295),
    "int64": (-9223372036854775808, 9223372036854775807),
    "uint64": (0, 18446744073709551615),
    "float16": (-65504.0, 65504.0),
    "float32": (-3.4028234663852886e+38, 3.4028234663852886e+38)
}


def _reduce_x_get_axis(context: OpInfo):
    axis = context.param.get("axes")  # reduce_x
    if axis is None:
        axis = context.param.get("axis")
    return axis


def __eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis]))
    return axis


def _reduce_sum(context: OpInfo):
    x = context.param.get("input_arrays")[0]
    axis = _reduce_x_get_axis(context)
    if not axis:
        noop_with_empty_axes = context.param.get("noop_with_empty_axes")
        if noop_with_empty_axes is None or noop_with_empty_axes:
            axis = []
        else:
            axis = []
            for i, _ in enumerate(x.shape):
                axis.append(i)

    axis = __eliminate_duplicate_axes(axis, context.param.get("input_arrays")[0])

    if len(axis) == 0:
        return x
    x_dtype = context.param.get("stc_input_dtypes")[0]
    y_dtype = context.param.get("output_dtypes")[0]
    if x_dtype in ("float16", "bfloat16"):
        x = x.astype(numpy.float32)
        y = numpy.sum(x, axis=axis)
    else:
        y = numpy.sum(x, axis=axis)
    if y_dtype == "bfloat16":
        return y.astype(tf.bfloat16.as_numpy_dtype, copy=False)
    return y.astype(y_dtype, copy=False)


def _reduce_mean(context: OpInfo):
    axis = context.param.get("axes")
    if axis is None:
        axis = context.param.get("axis")
    axis = __eliminate_duplicate_axes(axis, context.param.get("input_arrays")[0])
    x = context.param.get("input_arrays")[0]
    input_dtype = context.param.get("stc_input_dtypes")[0]
    output_dtype = context.param.get("output_dtypes")
    if len(axis) == 0:
        return x
    reduce_shape = [x.shape[idx] for idx, _ in enumerate(x.shape) if idx in axis]
    cof_value = functools.reduce(lambda x, y: x*y, reduce_shape)
    if context.param.get("stc_input_formats")[0] == "NC1HWC0" and (1 in axis) and (4 in axis):
        input_c_values = context.param.get("!input_c_values")
        if not input_c_values:
            c_index = context.param.get("stc_input_ori_formats")[0].index("C")
            input_c_values = [context.param.get("stc_ori_inputs")[0][c_index]]
        cof_value = cof_value/x.shape[1]/x.shape[4]*input_c_values[0]
    if input_dtype in ("float16", "bfloat16"):
        x = x.astype(numpy.float32)/cof_value
        y = numpy.sum(x, axis=axis)
        if output_dtype[0] == "float16":
            y = y.astype(numpy.float16)
        elif output_dtype[0] == "bfloat16":
            output_dtype = bfloat16_conversion(output_dtype)
            y = y.astype(output_dtype[0])
        return y
    else:
        x = x/cof_value
        return numpy.sum(x, axis=axis)