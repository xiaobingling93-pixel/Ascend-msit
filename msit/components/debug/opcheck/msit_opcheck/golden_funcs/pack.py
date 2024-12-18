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
from msit_opcheck.conversion.shape_convert import update_axis_for_npu_inner_format


def pack(context: OpInfo):
    ori_shape = context.param.get("stc_ori_inputs")[0]
    ori_format = context.param.get("stc_input_ori_formats")[0]
    input_format = context.param.get("dyn_input_formats")[0]
    axis = context.param.get("axis", 0)
    input_arrays = context.param.get("input_arrays")
    stack_axis = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
    return np.stack(input_arrays, axis=stack_axis)