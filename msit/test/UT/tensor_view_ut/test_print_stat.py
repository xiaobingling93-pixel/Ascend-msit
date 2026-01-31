# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import unittest
from io import StringIO
import logging
import torch

from components.utils.log import logger
from components.tensor_view.ait_tensor_view.print_stat import print_stat


class TestPrintStat(unittest.TestCase):
    def setUp(self):
        # 创建一个StringIO对象来捕获输出
        self.captured_output = StringIO()
        # 配置日志记录器，将其输出重定向到StringIO对象
        logger.setLevel(logging.INFO)
        self.stream_handler = logging.StreamHandler(self.captured_output)
        formatter = logging.Formatter('%(message)s')
        self.stream_handler.setFormatter(formatter)
        logger.addHandler(self.stream_handler)

    def tearDown(self):
        # 从日志记录器中移除stream_handler，避免影响其他测试
        logger.removeHandler(self.stream_handler)

    def test_valid_dtype(self):
        # 创建一个测试张量（有效的数据类型）
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        print_stat(tensor)
        log_content = self.captured_output.getvalue()
        self.assertIn("min", log_content)
        self.assertIn("max", log_content)
        self.assertIn("mean", log_content)
        self.assertIn("std", log_content)
        self.assertIn("var", log_content)

    def test_invalid_dtype(self):
        # 创建一个测试张量（无效的数据类型）
        tensor = torch.tensor([1, 2, 3], dtype=torch.int)
        print_stat(tensor)
        log_content = self.captured_output.getvalue()
        self.assertIn("min", log_content)
        self.assertIn("max", log_content)
        self.assertIn("mean", log_content)
        self.assertIn("std", log_content)
        self.assertIn("var", log_content)

    def test_tensor_with_nan(self):
        # 创建一个包含NaN值的张量
        tensor = torch.tensor([1.0, float('nan'), 3.0], dtype=torch.float32)
        print_stat(tensor)
        log_content = self.captured_output.getvalue()
        self.assertIn("nan", log_content)

    def test_large_tensor(self):
        # 创建一个大张量
        tensor = torch.randn(10000)
        print_stat(tensor)
        log_content = self.captured_output.getvalue()
        self.assertIn("min", log_content)
        self.assertIn("max", log_content)
        self.assertIn("mean", log_content)
        self.assertIn("std", log_content)
        self.assertIn("var", log_content)


if __name__ == '__main__':
    unittest.main()