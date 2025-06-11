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
import shutil
from unittest import TestCase
import ast
import pandas as pd
from st.utils import execute_cmd, check_column_actual, check_no_empty_lines_before_first_line
from st.utils import check_no_empty_lines_between_first_last_line, check_during_time


def check_split_csv_content(output_path, csv_file_name):
    # 校验该路径下是否正确生成csv文件，以及文件内容
    csv_file = os.path.join(output_path, csv_file_name)
    assert os.path.exists(csv_file)
    task_name = os.path.splitext(csv_file_name)[0]
    expected_header = ['name', 'during_time(millisecond)', 'max', 'min', 'mean', 'std', \
                       'pid', 'tid', 'start_time(microsecond)', 'end_time(microsecond)']
    if task_name == 'prefill':
        expected_header.append('rid')
    df = pd.read_csv(csv_file)
    # 检查列名是否正确
    result = check_column_actual(df.columns.tolist(), expected_header, context=csv_file_name)
    assert result, f"{csv_file_name} check column failed"
    # 检查是否存在空行
    check_no_empty_lines_before_first_line(df, context=csv_file_name)
    check_no_empty_lines_between_first_last_line(df, context=csv_file_name)
    # 检查执行时间是否正确
    result = check_during_time(df, context=csv_file_name)
    assert result, f"{csv_file_name} check during time failed"
   

class TestAnalyzeCmd(TestCase):
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER",
                             "/data/ms_service_profiler")
    INPUT_PATH = os.path.join(ST_DATA_PATH, "input/analyze/0506-1422")
    PREFILL_INPUT_PATH = os.path.join(ST_DATA_PATH, "input/analyze/PD_separate/prefill/3584-1024")
    DECODE_INPUT_PATH = os.path.join(ST_DATA_PATH, "input/analyze/PD_separate/decode/3584-1024")
    OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/split")
    PREFILL_CSV = "prefill.csv"
    DECODE_CSV = "decode.csv"
    COMMAND_SUCCESS = 0
    SPLIT_PROFILER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                    "ms_service_profiler_ext/split.py")
    COMMON_BATCH_SIZE = '1'
    PREFILL_BATCH_SIZE = '2'
    DECODE_BATCH_SIZE = '16'
    COMMON_RID = '10'
    PREFILL_RID = '2221'
    DECODE_RID = '105'

    def setUp(self):
        os.makedirs(self.OUTPUT_PATH, mode=0o750, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.OUTPUT_PATH)

    def check_split_task(self, task_name, output_path, csv_file_name):
        with self.subTest(f"Check {csv_file_name} content"):
            try:
                check_split_csv_content(output_path, csv_file_name)
            except Exception as e:
                self.fail(f"{task_name}: 检查 {csv_file_name} 时发生异常: {e}")

    def test_split_by_batch_size(self):
        # PD竞争 根据batch_size拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER,
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH,
            "--prefill-batch-size", self.COMMON_BATCH_SIZE,
            "--decode-batch-size", self.COMMON_BATCH_SIZE,
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable split task by batch size failed.")
            
        self.check_split_task("test_split_by_batch_size", self.OUTPUT_PATH, self.PREFILL_CSV)
        self.check_split_task("test_split_by_batch_size", self.OUTPUT_PATH, self.DECODE_CSV)

    def test_split_by_rid(self):
        # PD竞争 根据rid拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER,
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH,
            "--prefill-rid", self.COMMON_RID,
            "--decode-rid", self.COMMON_RID,
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable split task by rid failed.")
            
        self.check_split_task("test_split_by_rid", self.OUTPUT_PATH, self.PREFILL_CSV)
        self.check_split_task("test_split_by_rid", self.OUTPUT_PATH, self.DECODE_CSV)

    def test_split_data_in_p_node_by_batch_size(self):
        # PD分离 P 节点根据batch_size拆解 校验输出文件及内容
        cmd = ["python", self.SPLIT_PROFILER, 
               "--input-path", self.PREFILL_INPUT_PATH,
               "--output-path", self.OUTPUT_PATH,
               "--prefill-batch-size", self.PREFILL_BATCH_SIZE]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable split task by batch size in P Node failed.")

        self.check_split_task("test_split_data_in_p_node_by_batch_size", self.OUTPUT_PATH, self.PREFILL_CSV)

    def test_split_data_in_p_node_by_rid(self):
        # PD分离 P 节点根据rid拆解 校验输出文件及内容
        cmd = ["python", self.SPLIT_PROFILER, 
               "--input-path", self.PREFILL_INPUT_PATH,
               "--output-path", self.OUTPUT_PATH,
               "--prefill-rid", self.PREFILL_RID]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable split task by rid in P Node failed.")

        self.check_split_task("test_split_data_in_d_node_by_rid", self.OUTPUT_PATH, self.PREFILL_CSV)

    def test_split_data_in_d_node_by_batch_size(self):
        # PD分离 P 节点根据batch_size拆解 校验输出文件及内容
        cmd = ["python", self.SPLIT_PROFILER, 
               "--input-path", self.DECODE_INPUT_PATH,
               "--output-path", self.OUTPUT_PATH,
               "--decode-batch-size", self.DECODE_BATCH_SIZE]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable split task by batch size in D Node failed.")

        self.check_split_task("test_split_data_in_d_node_by_batch_size", self.OUTPUT_PATH, self.DECODE_CSV)

    def test_split_data_in_d_node_by_rid(self):
        # PD分离 P 节点根据rid拆解 校验输出文件及内容
        cmd = ["python", self.SPLIT_PROFILER, 
               "--input-path", self.DECODE_INPUT_PATH,
               "--output-path", self.OUTPUT_PATH,
               "--decode-rid", self.DECODE_RID]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable split task by rid in D Node failed.")

        self.check_split_task("test_split_data_in_d_node_by_rid", self.OUTPUT_PATH, self.DECODE_CSV)