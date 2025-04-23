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
import unittest
from unittest.mock import patch, mock_open, MagicMock
import tempfile

import json
import numpy as np


with patch.dict("sys.modules", {
    "c2lb": MagicMock(),
    "c2lb_dynamic": MagicMock(),
    "speculative_moe": MagicMock()
}):
    from components.expert_load_balancing.elb.load_balancing import save_matrix_to_json, dump_tables


class TestSaveMatrixToJson(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = self.temp_dir.name
        self.valid_deployment_3d = [
            [[0], [1]],
            [[2], [3]], 
            [[4], [5]] 
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_3d_matrix(self):
        file_name = "valid_output"
        expected_json = {
            "moe_layer_count": 3,
            "layer_list": [
                {
                    "layer_id": 0,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": [0]},
                        {"device_id": 1, "device_expert": [1]}
                    ]
                },
                {
                    "layer_id": 1,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": [2]},
                        {"device_id": 1, "device_expert": [3]}
                    ]
                },
                {
                    "layer_id": 2,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": [4]},
                        {"device_id": 1, "device_expert": [5]}
                    ]
                }
            ]
        }

        save_matrix_to_json(self.output_path, file_name, self.valid_deployment_3d)
        
        output_file = os.path.join(self.output_path, f"{file_name}.json")
        with open(output_file, 'r') as f:
            actual_data = json.load(f)
        
        self.assertEqual(actual_data, expected_json)

    def test_invalid_2d_matrix(self):
        invalid_deployment = [
            [0, 1],
            [2, 3]
        ]
        with self.assertRaises(ValueError) as context:
            save_matrix_to_json(self.output_path, "invalid", invalid_deployment)
        self.assertIn("必须是三维数组", str(context.exception))

    def test_invalid_4d_matrix(self):
        invalid_deployment = [[[[0]]]]  
        with self.assertRaises(ValueError) as context:
            save_matrix_to_json(self.output_path, "invalid", invalid_deployment)
        self.assertIn("必须是三维数组", str(context.exception))

    def test_empty_matrix(self):
        with self.assertRaises(ValueError) as context:
            save_matrix_to_json(self.output_path, "empty", [])
        self.assertIn("三维数组", str(context.exception))


class TestDumpTables(unittest.TestCase):
    def setUp(self):
        """ 在每个测试开始前运行 """
        self.test_path = "test.json"  # 定义测试文件路径
        if os.path.exists(self.test_path):  # 如果文件已存在，先删除
            os.remove(self.test_path)

    def tearDown(self):
        """ 在每个测试结束后运行 """
        if os.path.exists(self.test_path):  # 如果文件存在，删除它
            os.remove(self.test_path)

    @patch('components.utils.file_open_check.ms_open', mock_open())
    def test_dump_tables_given_valid_input_when_processed_then_success(self):
        """ 测试正常输入情况 """
        test_d2e =  [
            np.array([[1, 2], [3, 4]], dtype=np.int32), 
            np.array([[5, 6], [7, 8]], dtype=np.float32) 
        ]
        test_path = self.test_path
        
        dump_tables(test_path, test_d2e, n_devices=2)
        
        expected_json = {
            "moe_layer_count": 2,
            "layer_list": [
                {
                    "layer_id": 0,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": 1},
                        {"device_id": 1, "device_expert": 2}
                    ]
                },
                {
                    "layer_id": 1,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": 7.0},
                        {"device_id": 1, "device_expert": 8.0}
                    ]
                }
            ]
        }
        
        with open(test_path, 'r') as f:
            written_data = json.load(f)
            self.assertEqual(written_data, expected_json)

    @patch('components.utils.file_open_check.ms_open', mock_open())
    def test_dump_tables_given_empty_layers_when_processed_then_success(self):
        """ 测试空层列表 """
        dump_tables(self.test_path, [], n_devices=2)
        with open(self.test_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["moe_layer_count"], 0)

    def test_dump_tables_given_mismatch_device_count_when_processed_then_fail(self):
        with self.assertRaises(IndexError):
            dump_tables(self.test_path, np.array([[[1]]]), n_devices=2)

    @patch('components.utils.file_open_check.ms_open', mock_open())
    def test_dump_tables_given_single_device_when_processed_then_success(self):
        dump_tables(self.test_path, np.array([[[1,2,3]]]), n_devices=1)
        with open(self.test_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["layer_list"][0]["device_count"], 1)

    @patch('components.utils.file_open_check.ms_open', mock_open())
    def test_dump_tables_given_irregular_shape_when_processed_then_success(self):
        test_data = [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6], [7, 8], [9, 10]], dtype=np.int32)  
        ]
        dump_tables(self.test_path, test_data, n_devices=2)
        with open(self.test_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data["layer_list"][1]["device_list"]), 2)

    def test_dump_tables_given_invalid_layer_index_when_processed_then_fail(self):
        with self.assertRaises(IndexError):
            invalid_data = [MagicMock()]
            invalid_data[0].__getitem__.side_effect = IndexError
            dump_tables(self.test_path, invalid_data, n_devices=1)