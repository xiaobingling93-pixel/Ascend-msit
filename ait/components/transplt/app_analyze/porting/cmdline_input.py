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

from app_analyze.porting.porting_input import IInput
from app_analyze.common.kit_config import KitConfig, ReporterType, ScannerType, BuildToolType
from app_analyze.utils.io_util import IOUtil


class CommandLineInput(IInput):
    """
    CommandLineInput对象表示用户的输入来自命令行
    """

    # 继承父类的 __slots__
    __slots__ = []

    def __init__(self, args=None):
        super().__init__(args)

    @staticmethod
    def _check_path(folder):
        if not os.path.exists(folder):
            raise ValueError("{} msit transplt: error: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    'The path %s does not exist or you do not '
                                    'have the permission to access the path. '
                                    % folder))
        elif not os.path.isdir(folder):
            raise ValueError("{} msit transplt: error: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    'The path %s is '
                                    'not directory. ' % folder))
        elif not os.access(folder, os.R_OK):
            raise ValueError("{} msit transplt: error: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    "Cannot access the file "
                                    "or directory: %s" % folder))
        elif not os.access(folder, os.X_OK):
            raise ValueError("{} msit transplt: error: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    "Cannot access the "
                                    "directory: %s" % folder))
        elif IOUtil.check_path_is_empty(folder):
            raise ValueError("{} msit transplt: error: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    'The directory %s '
                                    'is empty' % folder))

    def resolve_user_input(self):
        """解析来自命令行的用户输入"""
        self.get_source_directories()
        self._get_construct_tool()
        self._set_debug_switch()
        self._get_output_type()
        self._get_scanner_mode()
        self.set_scanner_type()

    def get_source_directories(self):
        if not self.args.source:
            raise ValueError('msit transplt: error: the following arguments are required: -s/--source')

        for folder in self.args.source.split(','):
            folder = folder.strip()
            folder.replace('\\', os.path.sep)
            folder = os.path.realpath(folder)
            self._check_path(folder)
            if not folder.endswith(os.path.sep):
                folder += os.path.sep
            self.directories.append(folder)

        self.directories = sorted(set(self.directories), key=self.directories.index)
        self.source_path = self.directories
        self.directories = IOUtil.remove_subdirectory(self.directories)

    def _get_construct_tool(self):
        """获取构建工具类型"""
        if not self.args.tools:
            self.args.tools = 'make'
        if self.args.tools not in KitConfig.VALID_CONSTRUCT_TOOLS:
            raise ValueError('{} msit transplt: error: construct tool {} is not supported. supported input are {}.'
                             .format(KitConfig.PORTING_CONTENT, self.args.tools,
                                     ' or '.join(KitConfig.VALID_CONSTRUCT_TOOLS)))
        self.construct_tool = self.args.tools

    def _get_scanner_mode(self):
        """获取扫描方式"""
        if not self.args.mode:
            self.args.mode = 'all'

        if self.args.mode not in KitConfig.VALID_SCANNER_MODE:
            raise ValueError('{} msit transplt: error: scanner mode {} is not supported. supported input are {}.'
                             .format(KitConfig.PORTING_CONTENT, self.args.mode,
                                     ' or '.join(KitConfig.VALID_SCANNER_MODE)))
        self.scanner_mode = self.args.mode

    def _set_debug_switch(self):
        """动态修改日志级别"""
        self.debug_switch = self.args.log_level

    def _get_output_type(self):
        """获取输出报告格式"""
        out_format = self.args.report_type.lower()
        if out_format not in KitConfig.VALID_REPORT_TYPE:
            raise ValueError('msit transplt: error: output type {} is not supported. '
                             'supported input is csv/json.'.format(self.args.report_type))

        if out_format == 'csv':
            self.report_type.append(ReporterType.CSV_REPORTER)
        if out_format == 'json':
            self.report_type.append(ReporterType.JSON_REPORTER)

    def set_scanner_type(self):
        if self.construct_tool == BuildToolType.CMAKE.value:
            self.scanner_type.append(ScannerType.CMAKE_SCANNER)
            self.scanner_type.append(ScannerType.CPP_SCANNER)
        if self.construct_tool == BuildToolType.PYTHON.value:
            self.scanner_type.append(ScannerType.PYTHON_SCANNER)
        else:
            NotImplementedError('need to implementation.')
