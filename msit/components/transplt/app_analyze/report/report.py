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

from app_analyze.common.kit_config import FileType


class Report:
    """
    Report类作为输出报告的抽象基类存在，仅定义必要的属性和方法接口。
    第一个版本只支持csv格式的输出，所以相关的操作
    都在csv_report模块中进行实现；后续增加新的输出格式时，
    需要增加新的输出格式模块，并在project模块中增加实例化
    具体子类的分支.
    """
    FILE_TYPE_KEY = {
        FileType.MAKEFILE: "make",
        FileType.CMAKE_LISTS: "cmake_lists",
    }
    C_LINES_KEY = ("c", "make", "cmake_lists",)

    def __init__(self, report_param):
        """Report实例初始化函数"""
        self.report_path = report_param['directory']
        self.report_time = report_param['project_time']
        self._format = None

    def set_format(self, fmt):
        self._format = fmt

    def initialize(self, project):
        """抽象基类不是先该方法，交由各个子类实现"""
        raise NotImplementedError('{} must implement initialize method!'
                                  .format(self.__class__))

    def generate(self, fmt=None):
        """抽象基类不实现该方法，交由各个子类实现"""
        raise NotImplementedError('{} must implement generate method!'
                                  .format(self.__class__))

    def generate_abnormal(self, message):
        """抽象基类不实现该方法，交由各个子类实现"""
        raise NotImplementedError('{} must implement generate_abnormal method!'
                                  .format(self.__class__))
