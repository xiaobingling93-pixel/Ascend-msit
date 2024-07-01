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
import argparse
import os
import re

from components.utils.file_open_check import FileStat
from app_analyze.utils.log_util import logger

HPP_EXT = ('.h', '.hxx', '.hpp')


class FileMatrix:
    """定义待扫描文件矩阵"""

    def __init__(self, inputs):
        self.inputs = inputs
        self.files = {
            "cpp_sources": {},  # 待扫描的C/C++源文件列表
            "hpp_sources": [],  # 头文件列表
            "include_path": set(),  # 全局-I路径集合
            "makefiles": [],  # 待扫描的makefile文件列表
            "cmakefiles": [],  # 待扫描的makefile文件列表
            "python_files": [],  # 待扫描的py文件列表
        }
        self.excluded_dir_list = ['cmake-build-debug']
        self.excluded_head_file_list = []  # 需要排除的头文件列表

    @staticmethod
    def _walk_error(message):
        """
        将文件读取失败的场景输出到终端和log
        """
        logger.error(message)

    @staticmethod
    def _check_makefile(file_name):
        """
        检查file_name是否是makefile文件
        :param file_name:
        :return:
        """
        make_names = ('makefile', 'gnumakefile')
        # "Makefile.g++_openmpi.20220217223008.bak.0"形如这种文件则为备份makefile文件需过滤
        bak_file_check = re.match(r"^(Makefile|makefile|config.mk)\S+?\d{14}\.bak\.0$", file_name)
        if bak_file_check:
            return False
        is_makefile = file_name.lower() in make_names
        # 识别非标准类型Makefile（以Makefile、makefile和config.mk为前缀的文件，
        # 如Makefile.g++_openmpi、config.mk.dist）
        is_non_standard_makefile = re.match(r'Makefile|makefile|config.mk', file_name)
        if is_makefile or is_non_standard_makefile:
            return True
        return False

    def setup_file_matrix(self):
        """
        根据传入的源文件目录先做一次文件的查找
        :return: 源代码文件列表和makefile文件列表
        """
        # 如果需要执行全量扫描，或者成功执行了makefile的外部扫描，
        # 则需要遍历所有文件添加对应的文件
        self._do_global_scan_files()

    def get_files(self):
        return self.files

    def _do_global_scan_files(self):
        """
        遍历所有文件添加对应的文件
        :return:
        """
        for working_dir in self.inputs.directories:
            if os.path.isfile(working_dir):
                self._add_file_matrix(working_dir, False)
            else:
                for root, _, files in os.walk(working_dir, onerror=self._walk_error):
                    ex_flag = list(set(root.split(os.path.sep)).intersection(set(self.excluded_dir_list)))
                    if ex_flag:
                        continue

                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        self._add_file_matrix(file_path, False)

        logger.debug("The makefile files identified in the %s project are: %s",
                     self.inputs.directories, self.files.get('makefiles'))
        logger.debug("The cmakefile files identified in the %s project are: %s",
                     self.inputs.directories, self.files.get('cmakefiles'))

    def _add_file_matrix(self, file_path, from_external_tool,
                         src_files_from_external_tool=None):
        """
        添加文件到各自的files中
        :param from_external_tool: 是否是从外部工具读入的path
        :param file_path: 文件的绝对路径
        :param src_files_from_external_tool: 外部解析源文件字典
        :return:
        """
        # 判断是否为文件，不为文件跳过
        if not os.path.isfile(file_path):
            return

        # 判断文件有没有权限，没权限跳过
        if not os.access(file_path, os.R_OK):
            logger.warning("[Errno 13] Permission denied: '%s'", file_path)
            return

        # 判断文件是否再需要排序的头文件列表里
        if file_path in self.excluded_head_file_list:
            return

        # 需要遍历目录下的所有.h文件
        hpp_ext = HPP_EXT
        if self.inputs.construct_tool == 'cmake':
            # cmake工具执行成功，需要遍历目录下的所有.h文件
            hpp_ext = HPP_EXT

        # 判断是否为C/C++源码
        _, ext = os.path.splitext(file_path)
        if ext.lower() in hpp_ext:
            self.files.setdefault('hpp_sources', []).append(file_path)
            include_path = os.path.split(file_path)[0]
            self.files.setdefault('include_path', set()).add(include_path)

        self._check_all_file_type(
            file_path, ext, from_external_tool, src_files_from_external_tool)

    def _check_all_file_type(self, file_path, ext, from_external_tool,
                             src_files_from_external_tool):
        """
        检查全部文件类型：C/C++、python源文件、makefile文件
        """
        try:
            file_stat = FileStat(file_path)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"file path:{file_path} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read'):
            raise argparse.ArgumentTypeError(f"file path:{file_path} is illegal. Please check.")

        file_name = os.path.basename(file_path)
        source_ext = ('.c', '.cc', '.cpp', '.cxx', '.cx', '.cu')
        python_ext = ('.py', ".pyi")

        if ext.lower() in source_ext and not from_external_tool:
            self.files.setdefault('cpp_sources', {})[file_path] = ''
        if ext.lower() in source_ext and from_external_tool:
            self.files.setdefault('cpp_sources', {})[file_path] = \
                src_files_from_external_tool.get(file_path)
        if ext.lower() in python_ext and not from_external_tool:
            self.files.setdefault('python_files', {}).append(file_path)

        # 判断是否为Makefile
        if self._check_makefile(file_name):
            self.files.setdefault('makefiles', []).append(file_path)

        # 判断是否cmakelists.txt
        if file_name.lower() == 'cmakelists.txt':
            self.files.setdefault('cmakefiles', []).append(file_path)
