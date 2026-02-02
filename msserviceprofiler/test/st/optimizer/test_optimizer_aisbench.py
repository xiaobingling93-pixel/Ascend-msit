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
import glob
from unittest import TestCase
from msserviceprofiler.msguard.security import open_s


def check_csv_no_empty_start(csv_file):
    with open_s(csv_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行（第一行）

        for _, line in enumerate(f, start=2):  # 从第2行开始检查（数据行）
            stripped_line = line.strip()
            
            # 检查是否以逗号开头（即第一个单元格是否为空）
            if stripped_line.startswith(','):
                return False

        return True


def get_modelevalstate_path():
    # optimizer命令当前只能在该目录下运行
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    target_path = os.path.abspath(os.path.join(dir_path, "../../../msserviceprofiler/modelevalstate"))
    return target_path


class TestTrainCmd(TestCase):
    COMMAND_SUCCESS = 0
    MODELEVALSTATE_DIR = get_modelevalstate_path()
    RESULT_DIR = os.path.join(MODELEVALSTATE_DIR, "result")

    def setUp(self):
        if os.path.exists(self.RESULT_DIR):
            shutil.rmtree(self.RESULT_DIR)

    def tearDown(self):
        if os.path.exists(self.RESULT_DIR):
            shutil.rmtree(self.RESULT_DIR)

    def test_compare_ms_service_profiler_data(self):
        train_cmd = "cd " + self.MODELEVALSTATE_DIR + " && msserviceprofiler optimizer -b aisbench"

        self.assertEqual(os.system(train_cmd), self.COMMAND_SUCCESS)
        self.assertTrue(os.path.exists(self.RESULT_DIR))

        pattern = os.path.join(self.RESULT_DIR, "store", "data_storage_*.csv")
        matched_files = glob.glob(pattern)

        self.assertEqual(len(matched_files), 1)

        csv_file = matched_files[0]  # 取第一个匹配的文件
        result = check_csv_no_empty_start(csv_file)  # 调用检查函数
        self.assertTrue(result)
