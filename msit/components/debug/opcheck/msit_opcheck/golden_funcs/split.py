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

from msit_opcheck.graph_parser import OpInfo


def normalize_axis(axis, shape_length, ori_format, format) -> int:
    axis = axis if axis >= 0 else shape_length + axis

    format_map = {
        ("NHWC", "NC1HWC0"): {3: 1, 1: 2, 2: 3},
        ("NDHWC", "NDC1HWC0"): {4: 2, 2: 3, 3: 4},
        ("NCDHW", "NDC1HWC0"): {2: 1, 1: 2}
    }

    if (ori_format, format) in format_map:
        axis = format_map[(ori_format, format)].get(axis, axis)

    return axis


def _split_generic(context: OpInfo, split_type: str):
    input0, = context.param.get("input_arrays")
    split_dim = context.param.get("split_dim")
    stc_input_ori_format = context.param.get("stc_input_ori_formats")[0]
    stc_input_format = context.param.get("stc_input_formats")[0]
    split_dim = normalize_axis(split_dim, len(context.param.get("stc_ori_inputs")[0]), stc_input_ori_format,
                               stc_input_format)

    if split_type == 'v':
        size_splits = context.param.get("size_splits")
        indices = []
        start = 0
        for i in size_splits:
            start += i
            indices.append(start)
        return np.split(input0, indices[:-1], axis=split_dim)

    else:
        num_split = context.param.get("num_split")
        return np.split(input0, num_split, axis=split_dim)


def _split_v_d(context: OpInfo):
    res = _split_generic(context,split_type='v')
    return res


def _split(context: OpInfo):
    res = _split_generic(context, split_type='num')
    return res


def _split_d(context: OpInfo):
    res = _split_generic(context, split_type='num')
    return res