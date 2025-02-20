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

from ms_service_profiler.utils.log import logger


class FileCollector(object):
    def __init__(self, pattern: re.Pattern, max_iter=100) -> None:
        self.pattern = pattern
        self.max_iter = max_iter

        self._validate_param()

    def collect_pairs(self, dir_path_a: str, dir_path_b: str) -> List[Tuple]:
        file_set_a = self._collect(dir_path_a)
        file_set_b = self._collect(dir_path_b)

        intersection = file_set_a & file_set_b
        diff = file_set_a ^ file_set_b
        if diff:
            logger.error(
                "The files shown below are not matched in both directories %r and will not be compared", 
                list(diff)
            )

        return [
            (os.path.join(dir_path_a, file_path), os.path.join(dir_path_b, file_path))
            for file_path in intersection
        ]

    def _validate_param(self):
        if not isinstance(self.pattern, re.Pattern):
            raise ValueError("`pattern` type should be `re.Pattern`, but got %r instead" % type(self.pattern))

        if not isinstance(self.max_iter, int):
            raise ValueError("`max_iter` type should be `int`, but got %r instead" % type(self.max_iter))

    def _collect(self, dir_path: str) -> Set:
        res = set()

        files = os.listdir(dir_path)

        if len(files) > self.max_iter:
            raise RuntimeError("The number of the files under %r exceeds the iteration limits, "
                               "please use another directory instead" % dir_path)

        for file_path in files:
            if self.pattern.match(file_path):
                full_path = os.path.join(dir_path, file_path)
                
                if os.path.islink(full_path):
                    logger.warning("%r is a soft link and will not be compared, "
                                   "if this is not what you want, please use a regular file instead", file_path)
                    continue
                
                if not os.path.isfile(full_path):
                    logger.warning("%r is not a regular file and will not be compared, "
                                   "if this is not what you want, please use a regular file instead", file_path)
                    continue
                
                if os.path.getsize(full_path) > 10 * 1024 * 1024 * 1024:
                    logger.warning("%r exceeds the expected file size and will not be compared", file_path)
                    continue
                
                res.add(file_path)

        return res