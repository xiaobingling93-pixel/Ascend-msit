# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import array

import torch

from msit_llm.common.constant import ATTR_END, ATTR_OBJECT_LENGTH
from msit_llm.common.log import logger
from msit_llm.common.utils import check_input_path_legality, check_data_file_size


class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype = 0
        self.format = 0
        self.dims = []
        self.dtype_dict = {0: torch.float32, 1: torch.float16, 2: torch.int8, 3: torch.int32, 9: torch.int64,
                           12: torch.bool, 27: torch.bfloat16}

        self._parse_bin_file()

    def get_data(self):
        if self.dtype not in self.dtype_dict:
            logger.error("Unsupported dtype %s", self.dtype)
            raise ValueError("Unsupported dtype {}".format(self.dtype))
        dtype = self.dtype_dict.get(self.dtype)
        tensor = torch.frombuffer(array.array('b', self.obj_buffer), dtype=dtype)
        tensor = tensor.view(self.dims)
        return tensor

    def _parse_bin_file(self):
        with open(self.file_path, "rb") as fd:
            file_data = fd.read()

        begin_offset = 0
        for i, byte in enumerate(file_data):
            if byte == ord("\n"):
                line = file_data[begin_offset: i].decode("utf-8")
                begin_offset = i + 1
                fields = line.split("=")
                attr_name = fields[0]
                attr_value = fields[1]
                if attr_name == ATTR_END:
                    self.obj_buffer = file_data[i + 1:]
                    break
                elif attr_name.startswith("$"):
                    self._parse_system_attr(attr_name, attr_value)
                else:
                    self._parse_user_attr(attr_name, attr_value)

    def _parse_system_attr(self, attr_name, attr_value):
        if attr_name == ATTR_OBJECT_LENGTH:
            self.obj_len = int(attr_value)

    def _parse_user_attr(self, attr_name, attr_value):
        if attr_name == "dtype":
            self.dtype = int(attr_value)
        elif attr_name == "format":
            self.format = int(attr_value)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            self.dims = [int(dim) for dim in self.dims]


def read_atb_data(file_path):
    file_path = check_input_path_legality(file_path)

    if file_path.endswith(".bin"):
        if check_data_file_size(file_path):
            bin_tensor = TensorBinFile(file_path)
            data = bin_tensor.get_data()
            return data

    raise ValueError("Tensor file path must be end with .bin.")
