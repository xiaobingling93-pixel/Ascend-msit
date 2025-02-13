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

import glob
import os
import logging
import json
import shutil
import sqlite3
import ast
from unittest import TestCase
import ast
import pytest
import pandas as pd
from jsonschema import validate, ValidationError
from ...st.utils import execute_cmd


def check_column_actual(actual_columns, expected_columns, context):
    """检查实际列名是否与预期列名一致"""
    for col in expected_columns:
        assert col in actual_columns, f"在 {context} 中未找到预期列名: {col}"


def check_row(df, row_index, column):
    """检查指定行和列的数据是否为数字"""
    try:
        value = df.at[row_index, column]
        # 尝试将值转换为数字
        float(value)
    except (ValueError, KeyError):
        assert False, f"在 {column} 列的第 {row_index} 行，值 {value} 不是有效的数字"


def check_summary_csv_content(output_path, csv_file_name):
    csv_file = os.path.join(output_path, csv_file_name)
    # 检查文件是否存在
    assert os.path.exists(csv_file), f"文件 {csv_file} 不存在"
    assert os.path.isfile(csv_file), f"{csv_file} 不是一个有效的文件"
    df = pd.read_csv(csv_file)
    actual_columns = df.columns.tolist()

    expected_csv_columns = ['Metric', 'average', 'max', 'min', 'P50', 'P90', 'P99']
    check_column_actual(actual_columns, expected_csv_columns, context=csv_file_name)

    columns_to_check = ['average', 'max', 'min', 'P50', 'P90', 'P99']

    # 检查列名是否存在
    for column in columns_to_check:
        assert column in actual_columns, f"在 {csv_file_name} 中未找到预期列名: {column}"

    # 新增：检查所有行的特定列数据格式是否为数字
    for column in columns_to_check:
        for index, value in df[column].items():
            try:
                float(value)
            except ValueError:
                assert False, f"在 {column} 列的第 {index} 行，值 {value} 不是有效的数字"
    return True


def check_service_summary_csv_content(output_path, csv_file_name):
    csv_file = os.path.join(output_path, csv_file_name)
    # 检查文件是否存在
    assert os.path.exists(csv_file), f"文件 {csv_file} 不存在"
    assert os.path.isfile(csv_file), f"{csv_file} 不是一个有效的文件"
    df = pd.read_csv(csv_file)
    actual_columns = df.columns.tolist()

    expected_csv_columns = ['Metric', 'value']
    check_column_actual(actual_columns, expected_csv_columns, context=csv_file_name)

    columns_to_check = ['value']

    # 检查列名是否存在
    for column in columns_to_check:
        assert column in actual_columns, f"在 {csv_file_name} 中未找到预期列名: {column}"

    # 新增：检查所有行的特定列数据格式是否为数字
    for column in columns_to_check:
        for index, value in df[column].items():
            try:
                float(value)
            except ValueError:
                assert False, f"在 {column} 列的第 {index} 行，值 {value} 不是有效的数字"
    return True


class TestAnalyzeCmd(TestCase):
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER", "/data/ms_service_profiler")
    INPUT_PATH = os.path.join(ST_DATA_PATH, "input/analyze/0211-1226")
    OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/analyze")
    REQUEST_CSV_FILE_NAME = "request_summary.csv"
    BATCH_CSV_FILE_NAME = "batch_summary.csv"
    SERVICE_CSV_FILE_NAME = "service_summary.csv"
    COMMAND_SUCCESS = 0
    ANALYZE_PROFILER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                    "ms_service_profiler_ext/analyze.py")

    def setup_class(self):
        os.makedirs(self.OUTPUT_PATH, mode=0o750, exist_ok=True)

    def teardown_class(self):
        shutil.rmtree(self.OUTPUT_PATH)

    def test_analyze_ms_service_profiler_data(self):
        # 校验msserviceprofiler打点采集数据解析功能是否正常解析，校验输出文件及内容
        cmd = [
            "python", self.ANALYZE_PROFILER,
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable ms service profiler analyze task failed.")

        # 校验summary.csv生成
        with self.subTest("Check request_summary.csv content"):
            check_summary_csv_content(self.OUTPUT_PATH, self.REQUEST_CSV_FILE_NAME)
        with self.subTest("Check batch_summary.csv content"):
            check_summary_csv_content(self.OUTPUT_PATH, self.BATCH_CSV_FILE_NAME)
        with self.subTest("Check service_summary.csv content"):
            check_service_summary_csv_content(self.OUTPUT_PATH, self.SERVICE_CSV_FILE_NAME)