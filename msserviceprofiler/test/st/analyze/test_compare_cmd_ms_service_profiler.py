# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import shutil
from pathlib import Path
from unittest import TestCase

from st.utils import execute_cmd


class TestCompareCmd(TestCase):
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER",
                             "/data/ms_service_profiler")
    INPUT_PATH = os.path.join(ST_DATA_PATH, "input/analyze/1225-196-10Req")
    ANALYZE_OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/analyze")
    COMPARE_OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/compare")
    COMMAND_SUCCESS = 0
    ANALYZE_SCRIPT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                    "msserviceprofiler/__main__.py")
    COMPARE_SCRIPT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                    "msserviceprofiler/__main__.py")

    def setUp(self):
        os.makedirs(self.ANALYZE_OUTPUT_PATH, mode=0o750, exist_ok=True)
        os.makedirs(self.COMPARE_OUTPUT_PATH, mode=0o750, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.ANALYZE_OUTPUT_PATH)
        shutil.rmtree(self.COMPARE_OUTPUT_PATH)

    def test_compare_ms_service_profiler_data(self):
        # 校验msserviceprofiler打点采集数据解析功能是否正常解析，校验输出文件及内容
        analyze_cmd = [
            "python", self.ANALYZE_SCRIPT, "analyze",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.ANALYZE_OUTPUT_PATH
        ]
        if execute_cmd(analyze_cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.ANALYZE_OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable ms service profiler analyze task failed.")
            return
        
        compare_cmd = [
            "python", self.COMPARE_SCRIPT, "compare",
            self.ANALYZE_OUTPUT_PATH,
            self.ANALYZE_OUTPUT_PATH,
            "--output-path", self.COMPARE_OUTPUT_PATH
        ]
        if execute_cmd(compare_cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.COMPARE_OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable ms service profiler compare task failed.")
            return
        self.assertTrue((Path(self.COMPARE_OUTPUT_PATH) / 'compare_result.xlsx').exists())
        self.assertTrue((Path(self.COMPARE_OUTPUT_PATH) / 'compare_result.db').exists())
        self.assertTrue((Path(self.COMPARE_OUTPUT_PATH) / 'compare_visualization.json').exists())
