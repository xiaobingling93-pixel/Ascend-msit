# -*- coding: utf-8 -*-
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

import os
import re
from typing import List, Tuple, Set


class FileCollector(object):
    def __init__(self, pattern: re.Pattern, max_iter=100) -> None:
        self.pattern = pattern
        self.max_iter = max_iter

        self._validate_param()

    def collect_pairs(self, dir_path_a: str, dir_path_b: str) -> List[Tuple]:
        file_set_a = self._collect(dir_path_a)
        file_set_b = self._collect(dir_path_b)

        intersection = file_set_a & file_set_b

        return [
            (os.path.join(dir_path_a, file_path), os.path.join(dir_path_b, file_path))
            for file_path in intersection
        ]

    def _validate_param(self):
        if not isinstance(self.pattern, re.Pattern):
            raise ValueError

        if not isinstance(self.max_iter, int):
            raise ValueError

    def _collect(self, dir_path: str) -> Set:
        res = set()

        files = os.listdir(dir_path)

        if len(files) > self.max_iter:
            raise RuntimeError

        for file_path in files:
            if self.pattern.match(file_path):
                res.add(file_path)

        return res
