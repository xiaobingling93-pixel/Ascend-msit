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

from collections import deque

import torch
import numpy as np

from components.debug.opcheck.graph_parser import OpInfo
from components.debug.opcheck.utils import get

def _pad(context: OpInfo):
    bf16_mark = False
    x = context.param.get("input_arrays")[0]
    paddings = context.param.get("paddings")
    x_format = get(context.param.get("stc_input_formats", 0))

    pad_shape = deque()
    for i in range(len(paddings)):
        pad_shape.append(paddings[len(paddings) - 1 - i][0])
        pad_shape.append(paddings[len(paddings) - 1 - i][1])
    pad_shape_tensor = torch.tensor(pad_shape)

    if (x_format == "NC1HWC0"):
        pad_shape.appendleft(0)
        pad_shape.appendleft(0)
    if "bfloat16" in str(x.dtype):
        bf16_mark = True
        x = x.astype(np.float32)
    x_tensor = torch.from_numpy(x)
    golden = torch.constant_pad_nd(x_tensor, tuple(pad_shape), 0)
    if bf16_mark:
        golden.to(torch.bfloat16)
    res = golden.numpy()
    return res
