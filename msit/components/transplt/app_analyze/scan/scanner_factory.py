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

from app_analyze.common.kit_config import ScannerType
from app_analyze.scan.cxx_scanner import CxxScanner
from app_analyze.scan.cmake_scanner import CMakeScanner
from app_analyze.scan.python_scanner import PythonScanner


def merge_dicts(*dict_args):
    """ 字典合并 """
    result = {}
    for item in dict_args:
        result.update(item)

    return result


class ScannerFactory:
    """
    扫描器工厂类
    """

    def __init__(self, scanner_params):
        """实例化扫描器工厂对象"""
        self.scanner_params = scanner_params

    def get_scanner(self, scanner_type, project_directory=None):
        """
        工厂生产方法
        :param scanner_type: 扫描器种类
        :return: 扫描器实例对象
        """
        if scanner_type == ScannerType.CPP_SCANNER:
            cxx_files = list(self.scanner_params['cpp_files']['cpp'].keys())
            cxx_parser = self.scanner_params['cpp_files']['cxx_parser']
            return CxxScanner(cxx_files, cxx_parser)
        if scanner_type == ScannerType.CMAKE_SCANNER:
            return CMakeScanner(self.scanner_params['cmake_files'])
        if scanner_type == ScannerType.PYTHON_SCANNER:
            return PythonScanner(self.scanner_params['python_files'], project_directory)
        raise Exception('Impossible Scanner Type!')
