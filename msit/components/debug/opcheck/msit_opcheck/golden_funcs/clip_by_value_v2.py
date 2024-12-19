# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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


def _clip_by_value(context: OpInfo):
    is_v2 = context.param.get("op_name") == "clip_by_value_v2"
    input_t = context.param.get("input_arrays")[0]
    clip_value_min = context.param.get("input_arrays")[1]
    clip_value_max = context.param.get("input_arrays")[2]
    if "bfloat16" in str(context.param.get("input_arrays")[0].dtype):
        input_t = input_t.astype("float32")
        clip_value_min = clip_value_min.astype("float32")
        clip_value_max = clip_value_max.astype("float32")
    if is_v2:
        max_ = np.maximum(input_t, clip_value_min)
        res = np.minimum(max_, clip_value_max)
    else:
        min_ = np.minimum(input_t, clip_value_max)
        res = np.maximum(min_, clip_value_min)
    if "bfloat16" in str(context.param.get("input_arrays")[0].dtype):
        return res.astype(context.param.get("input_arrays")[0].dtype, copy=False)
    return res