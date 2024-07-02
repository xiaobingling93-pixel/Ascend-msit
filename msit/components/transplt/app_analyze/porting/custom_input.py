# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

from app_analyze.porting.cmdline_input import CommandLineInput
from app_analyze.common.kit_config import ScannerType


class CustomInput(CommandLineInput):
    """
    CommandLineInput对象表示用户的输入来自命令行
    """

    # 继承父类的 __slots__
    __slots__ = []

    def __init__(self, args=None):
        super().__init__(args)

    def get_source_directories(self):
        if not self.args.source:
            raise ValueError('No input files!')
        # 目录或单个文件
        folder = self.args.source.strip()
        folder.replace('\\', '/')
        folder = os.path.realpath(folder)
        if os.path.isdir(folder):
            self._check_path(folder)
            if not folder.endswith(os.path.sep):
                folder += os.path.sep

        self.directories.append(folder)
        self.source_path = self.directories

    def set_scanner_type(self):
        if self.construct_tool == "cmake":
            self.scanner_type.append(ScannerType.CPP_SCANNER)
        else:
            NotImplementedError('need to implementation.')
