# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from array import array

import torch
from torch import float16, float32, int8, int32, int64, bfloat16

ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"

dtype_dict = {
    0: float32,
    1: float16,
    2: int8,
    3: int32,
    9: int64,
    12: torch.bool,
    27: bfloat16
}


def read_atb_data(path: str) -> torch.Tensor:
    dtype = 0
    dims = []

    with open(path, "rb") as fd:
        file_data = fd.read()

    offset = 0
    obj_buffer = ()

    for i, byte in enumerate(file_data):
        if byte == ord("\n"):
            line = file_data[offset:i].decode("utf-8")
            offset = i + 1
            [attr_name, attr_value] = line.split("=")

            if attr_name == ATTR_END:
                obj_buffer = file_data[i + 1:]
                break
            elif attr_name.startswith("$"):
                pass
            else:
                if attr_name == "dtype":
                    dtype = int(attr_value)
                elif attr_name == "dims":
                    dims = [int(x) for x in attr_value.split(",")]

    if dtype not in dtype_dict:
        raise ValueError(f"Unsupported dtype: {dtype}")

    dtype = dtype_dict.get(dtype)
    tensor = torch.frombuffer(array("b", obj_buffer), dtype=dtype)

    return tensor.view(dims)


def write_atb_data(tensor: torch.Tensor, path: str):
    _dtype_map = {v: k for k, v in dtype_dict.items()}
    dtype = _dtype_map.get(tensor.dtype)
    dims = ','.join(map(str, tensor.shape))
    data = tensor.numpy().tobytes()

    meta = f"dtype={dtype}\ndims={dims}\n$End=1\n".encode("utf-8")

    with open(path, "wb") as fo:
        fo.write(meta + data)

    del _dtype_map
