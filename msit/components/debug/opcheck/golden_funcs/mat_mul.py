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
import torch

from components.debug.opcheck.graph_parser import OpInfo
from components.debug.opcheck.conversion.dtype_convert import get, bfloat16_conversion_v2
from components.debug.opcheck.utils import ceil_div, align


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def nd_to_fractal_nz(data: np.ndarray, trans_align_nd=False):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    if data.dtype == "int8":
        m0, n0 = 16, 32
    else:
        m0, n0 = 16, 16
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    if trans_align_nd:
        return data
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def hf_32_input_gerenate(context: OpInfo, input_fp32):
    input_hf32 = input_fp32.view(np.int32)
    input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
    input_hf32 = np.left_shift(input_hf32, 12)
    input_hf32 = input_hf32.view(np.float32)
    return input_hf32


def helper_mm_and_bmm(context: OpInfo, name_parameter):
    tf.compat.v1.disable_eager_execution()
    x1, x2, bias, *_ = context.param.get("input_arrays")
    trans_a = context.param.get(name_parameter[0])
    trans_b = context.param.get(name_parameter[1])
    format_bias = get(context.param.get("stc_input_formats"), 2)
    format_out = context.param.get("output_formats")[0]

    if context.param.get("impl_mode") == "enable_hi_float_32_execution":
        x1 = hf_32_input_gerenate(context, x1)
        x2 = hf_32_input_gerenate(context, x2)

    if x1.dtype == 'float32':
        a_data = x1.astype('float64')
        b_data = x2.astype('float64')
    else:
        a_data = x1.astype('float32')
        b_data = x2.astype('float32')

    a = torch.from_numpy(a_data)
    b = torch.from_numpy(b_data)
    if trans_a:
        if len(a.shape) == 2:
            a = a.t()
        else:
            a = a.transpose(-1,-2)
    if trans_b:
        if len(b.shape) == 2:
            b = b.t()
        else:
            b = b.transpose(-1,-2)
    res_pt = torch.matmul(a, b).numpy()
    if bias is not None:
        if x1.dtype == 'float32':
            bias = bias.astype('float64')
        else:
            bias = bias.astype('float32')
        if format_bias == 'NC1HWC0':
            res_pt = nd_to_fractal_nz(res_pt, True)
            bias = bias.transpose(0, 1, 4, 2, 3).reshape(bias.shape)
            res_pt = torch.from_numpy(res_pt)
            bias = torch.from_numpy(bias)
            res_pt = torch.add(res_pt, bias).numpy()
        else:
            if format_out == "FRACTAL_NZ":
                res_pt = nd_to_fractal_nz(res_pt, True)
                bias_shape = bias.shape[0]
                align_bias = align(bias_shape, 16)
                bias = np.pad(bias, (0, align_bias - bias_shape), 'constant', constant_values=(0, 0))
                res_pt = torch.from_numpy(res_pt)
                bias = torch.from_numpy(bias)
                res_pt = torch.add(res_pt, bias).numpy()
            else:
                res_pt = torch.from_numpy(res_pt)
                bias = torch.from_numpy(bias)
                res_pt = torch.add(res_pt, bias).numpy()

    if format_out == 'FRACTAL_NZ':
        res_pt = nd_to_fractal_nz(res_pt)
    output_dtype = bfloat16_conversion_v2(context.params.get("output_dtypes"))
    return res_pt.astype(output_dtype[0], copy=False)


def _matmul(context: OpInfo):
    return helper_mm_and_bmm(context, ["transpose_x1", "transpose_x2"])


