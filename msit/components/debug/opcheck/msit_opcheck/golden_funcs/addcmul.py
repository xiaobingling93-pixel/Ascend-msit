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
from msit_opcheck.utils import broadcast_to_maxshape


def _addcmul(context: OpInfo):
    input0, input1, input2, input3 = context.param.get("input_arrays")
    if context.param.get("output_dtypes")[0] == "float16":
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
        input2 = input2.astype("float32")
        input3 = input3.astype("float32")
    input_data_shape = input0.shape
    x1_shape = input1.shape
    x2_shape = input2.shape
    value_shape = input3.shape
    input_data_shape, x1_shape, x2_shape, value_shape, shape_max = \
        broadcast_to_maxshape([input_data_shape, x1_shape, x2_shape, value_shape])
    input_data = np.broadcast_to(input0, shape_max)
    x1 = np.broadcast_to(input1, shape_max)
    x2 = np.broadcast_to(input2, shape_max)
    value = np.broadcast_to(input3, shape_max)

    vmul_val = np.multiply(x1, x2)
    vmul_val2 = np.multiply(vmul_val, value)
    res = np.add(input_data, vmul_val2)

    return res.astype(context.param.get("output_dtypes")[0])
