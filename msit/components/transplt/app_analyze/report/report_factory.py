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

from app_analyze.common.kit_config import ReporterType
from app_analyze.report.csv_report import CsvReport
from app_analyze.report.json_report import JsonReport
from app_analyze.utils.log_util import logger


class ReporterFactory:
    """
    报告工厂类, 定义了输出文件后缀和具体处理类之间的对应关系
    """

    def __init__(self, report_params):
        """报告工厂类实例化函数"""
        self.report_params = report_params

    def get_reporter(self, report_type, info=None):
        """
        如果后面要增加新的输出报告格式，需要新增一个具体的报告子类，添加报告枚举，并在这里增加实例化的逻辑
        :param report_type: 报告枚举类型
        :param info: 任务基本信息
        :return: 报告实例对象
        """
        if report_type == ReporterType.CSV_REPORTER:
            return CsvReport(self.report_params)
        if report_type == ReporterType.JSON_REPORTER:
            return JsonReport(self.report_params)

        raise Exception('only support Csv/JSON report format.')

    def dump(self):
        """
        打印信息
        :return:NA
        """
        logger.debug(self.report_params)
