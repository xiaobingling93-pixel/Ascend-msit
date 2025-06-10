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
import logging
from abc import ABC, abstractmethod


class BaseReporter(ABC):
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        self.logger = self._init_logger()

    @staticmethod
    def _init_logger():
        local_logger = logging.getLogger(__name__)
        local_logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        stream_handler.setFormatter(formatter)
        local_logger.addHandler(stream_handler)
        return local_logger
    
    def print_title(self, title: str, fillchar):
        try:
            cols, _ = os.get_terminal_size()
        except OSError:
            cols = 80
        self.logger.info(f" {title} ".center(cols, fillchar))
    
    @abstractmethod
    def report(self, check_results):
        pass
