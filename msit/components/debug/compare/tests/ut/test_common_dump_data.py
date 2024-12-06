# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import time

from components.utils.check.rule import Rule
from components.debug.compare.msquickcmp.common.utils import AccuracyCompareException
from components.debug.compare.msquickcmp.common.dump_data import DumpData  # 假设 DumpData 类在 your_module.py 文件中


class TestDumpData(unittest.TestCase):

    def test_to_valid_name(self):
        # 测试 _to_valid_name 方法
        dump_data = DumpData()
        valid_name = dump_data._to_valid_name("model.name/with.slash")
        self.assertEqual(valid_name, "model_name_with_slash")

    @patch('os.path.exists')
    @patch('os.access')
    def test_check_path_exists_valid(self, mock_access, mock_exists):
        # 测试路径存在且可读
        mock_exists.return_value = True
        mock_access.return_value = True

        dump_data = DumpData()
        dump_data._check_path_exists("./mock/path.om", extentions=[".om"])

    @patch('os.path.exists')
    def test_check_path_exists_not_exists(self, mock_exists):
        # 测试路径不存在
        mock_exists.return_value = False
        with self.assertRaises(AccuracyCompareException):
            dump_data = DumpData()
            dump_data._check_path_exists("/mock/invalid_path", extentions=[".om"])

    @patch('os.path.exists')
    @patch('os.access')
    def test_check_path_exists_invalid_extension(self, mock_access, mock_exists):
        # 测试文件扩展名错误
        mock_exists.return_value = True
        mock_access.return_value = True
        with self.assertRaises(AccuracyCompareException):
            dump_data = DumpData()
            dump_data._check_path_exists("/mock/path.txt", extentions=[".om"])

    @patch('os.path.exists')
    @patch('os.access')
    def test_check_path_exists_no_read_permission(self, mock_access, mock_exists):
        # 测试没有读取权限
        mock_exists.return_value = True
        mock_access.return_value = False
        with self.assertRaises(AccuracyCompareException):
            dump_data = DumpData()
            dump_data._check_path_exists("/mock/path", extentions=[".om"])

    @patch('time.time', return_value=1627551000)
    def test_generate_dump_data_file_name(self, mock_time):
        # 测试生成文件名
        dump_data = DumpData()
        file_name = dump_data._generate_dump_data_file_name("model.name", 1)
        expected_file_name = "model_name.1.1627551000000000.npy"
        self.assertEqual(file_name, expected_file_name)

    def test_check_input_data_path_valid(self):
        # 测试输入数据路径检查有效
        dump_data = DumpData()
        input_paths = ["/mock/input1.bin", "/mock/input2.bin"]
        inputs_tensor_info = [{"name": "input1", "shape": (1, 1)}, {"name": "input2", "shape": (1, 1)}]

        with patch('os.path.exists', return_value=True):
            dump_data._check_input_data_path(input_paths, inputs_tensor_info)

    def test_check_input_data_path_invalid(self):
        # 测试输入数据路径检查无效（路径数目不匹配）
        dump_data = DumpData()
        input_paths = ["/mock/input1.bin"]
        inputs_tensor_info = [{"name": "input1", "shape": (1, 1)}, {"name": "input2", "shape": (1, 1)}]

        with self.assertRaises(AccuracyCompareException):
            dump_data._check_input_data_path(input_paths, inputs_tensor_info)

    @patch('os.path.exists')
    @patch('os.access')
    def test_generate_random_input_data(self, mock_access, mock_exists):
        # 测试生成随机输入数据并保存为文件
        mock_exists.return_value = True
        mock_access.return_value = True

        dump_data = DumpData()
        save_dir = "./mock/save_dir"
        names = ["input1", "input2"]
        shapes = [(1, 1), (1, 1)]
        dtypes = [np.float32, np.float32]

        with patch('numpy.random.random') as mock_random:
            mock_random.return_value = np.array([[1.0]], dtype=np.float32)
            inputs_map = dump_data._generate_random_input_data(save_dir, names, shapes, dtypes)

        self.assertIn("input1", inputs_map)
        self.assertIn("input2", inputs_map)

    @patch('os.path.exists')
    @patch('os.access')
    @patch('numpy.fromfile')
    def test_read_input_data(self, mock_fromfile, mock_access, mock_exists):
        # 测试读取输入数据并验证其形状
        mock_exists.return_value = True
        mock_access.return_value = True

        input_paths = ["./mock/input1.bin", "./mock/input2.bin"]
        names = ["input1", "input2"]
        shapes = [(1, 1), (1, 1)]
        dtypes = [np.float32, np.float32]

        mock_fromfile.return_value = np.array([1.0], dtype=np.float32)

        dump_data = DumpData()
        inputs_map = dump_data._read_input_data(input_paths, names, shapes, dtypes)

        self.assertIn("input1", inputs_map)
        self.assertIn("input2", inputs_map)
        self.assertEqual(inputs_map["input1"].shape, (1, 1))
        self.assertEqual(inputs_map["input2"].shape, (1, 1))