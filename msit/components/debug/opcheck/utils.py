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


def broadcast_to_maxshape(shapes: list):
    """
    produce broadcast shape
    for example:
        input: shape is [[2, 3], [3, 2, 1], [3, 1, 3]]
        output: [1, 2, 3], [3, 2, 1], [3, 1, 3], [3, 2, 3]
    """
    def _max(_shape):
        no_one_shape = [s for s in _shape if s != 1]
        if len(no_one_shape) == 0:
            max_value = 1
        else:
            max_value = no_one_shape[0]
        return max_value
    max_dim_length = max(len(list(shape)) for shape in shapes)
    input_shapes = []
    for shape in shapes:
        input_shapes.append([1 for _ in range(max_dim_length - len(shape))] + list(shape))
    input_shapes = list(map(list, zip(*input_shapes)))
    max_shape = [_max(shape) for shape in input_shapes]
    input_shapes = list(map(list, zip(*input_shapes)))
    return (*input_shapes, max_shape)


def compare_value_int32(data_x, data_y, shape_dz):
    min_value_int = np.array(1, dtype="int32")
    data_zero_int = np.array(0, dtype="int32")
    min_value_tensor = np.broadcast_to(min_value_int, shape_dz)
    data_zero_int_tensor = np.broadcast_to(data_zero_int, shape_dz)
    sub_xy = np.subtract(data_x, data_y)
    add_min = np.add(sub_xy, min_value_tensor)
    vmax_zero = np.maximum(add_min, data_zero_int_tensor)
    return np.minimum(vmax_zero, min_value_tensor)


def compare_value_float(data_x, data_y):
    min_value = np.array(2 ** (-126), dtype="float32")
    max_value = np.array(2 ** 62, dtype="float32")
    max_value_1 = np.array(2 ** 2, dtype="float32")
    data_zero = np.multiply(data_x, 0)
    min_value_tensor = np.add(data_zero, min_value)
    max_value_tensor = np.add(data_zero, max_value)
    max_value_1_tensor = np.add(data_zero, max_value_1)
    sub_xy = np.subtract(data_x, data_y)
    add_min_value = np.add(sub_xy, min_value)
    vmax_zero = np.maximum(add_min_value, data_zero)
    vmin_min_value = np.minimum(vmax_zero, min_value_tensor)
    vmul_max_value = np.multiply(vmin_min_value, max_value_tensor)
    vmul_max_value_1 = np.multiply(vmul_max_value, max_value_tensor)
    return np.multiply(vmul_max_value_1, max_value_1_tensor)


def compare_value(data_x, data_y, dtype, shape_dz):
    if dtype == "int32":
        return compare_value_int32(data_x, data_y, shape_dz)
    else:
        return compare_value_float(data_x, data_y)


def calculate_result_le(data_x, data_y, data_dz, dtype, shape_dz):
    minus_one = np.array(-1, dtype="float32")
    value_one = np.array(1, dtype="float32")
    if dtype == "int32":
        minus_one = np.array(-1, dtype="int32")
        value_one = np.array(1, dtype="int32")
    minus_one_tensor = np.broadcast_to(minus_one, shape_dz)
    value_one_tensor = np.broadcast_to(value_one, shape_dz)
    datax_select_le = compare_value(data_y, data_x, dtype, shape_dz)
    result_dx = np.multiply(data_dz, datax_select_le)
    select_reverse = np.subtract(datax_select_le, value_one_tensor)
    select_dy = np.multiply(select_reverse, minus_one_tensor)
    result_dy = np.multiply(data_dz, select_dy)
    return result_dx, result_dy


def ceil_div(a, b):
    return (a + b -1) // b


def align(a, b):
    return ceil_div(a, b) * b


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    data = np.nan_to_num(data)
    return data
