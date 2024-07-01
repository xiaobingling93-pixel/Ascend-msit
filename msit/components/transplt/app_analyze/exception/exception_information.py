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

from enum import Enum, unique


@unique
class SourceScanErrorCode(Enum):
    """
    定义底层源码扫描与服务端代码的错误码
    """
    cmake_execute_failed = '2001'
    makefile_execute_failed = '2002'
    automake_execute_failed = '2003'
    source_scan_no_result = '2004'
    source_file_not_found = '2005'


@unique
class SourceScanErrorInfo(Enum):
    """
    源码扫描错误信息
    """
    source_scan_error_info = {
        '2001': {
            'cn': 'Cmake执行失败：',
            'en': 'Cmake execute failed: '
        },
        '2002': {
            'cn': 'Makefile执行失败：',
            'en': 'Makefile execute failed: '
        },
        '2003': {
            'cn': 'Automake执行失败：',
            'en': 'Automake execute failed: '
        },
        '2004': {
            'cn': '源码扫描没有结果',
            'en': 'Source scan with no results.'
        },
        '2005': {
            'cn': '源码文件丢失',
            'en': 'Source code not found.'
        }
    }

    @staticmethod
    def get_en_info(error_code):
        """ 获取源码扫描错误信息 """
        error_info = SourceScanErrorInfo.source_scan_error_info.get(
            error_code)
        if error_info:
            return error_info.get('en')
        else:
            return 'error not specified'

    @staticmethod
    def get_cn_info(error_code):
        """ 获取源码扫描错误信息 """
        error_info = SourceScanErrorInfo.source_scan_error_info.get(
            error_code)
        if error_info:
            return error_info.get('cn')
        else:
            return 'error not specified'
