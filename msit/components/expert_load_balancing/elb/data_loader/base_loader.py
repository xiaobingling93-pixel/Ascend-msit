# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
from enum import Enum


class DataType(Enum):
    MINDIE_SUMED_CSV = 0
    MINDIE_SPLITED_CSV = 1
    MINDIE_SPLITED_CSV_WITH_TOPK = 2
    VLLM_SUMED_TENSOR = 3
    UNKNOWN_TYPE = 999


class BaseDataLoader:
    def __init__(self, input_args):
        self.input_args = input_args
        self.data_type = DataType.UNKNOWN_TYPE

    @staticmethod
    def load_from_file(file_path):
        return None
