import shutil
import unittest
import tempfile
import re
import pytest
from unittest.mock import patch, mock_open, MagicMock, call
import os

import numpy as np
import pandas as pd

from components.utils.file_open_check import OpenException
from components.utils.log import logger


with patch.dict("sys.modules", {
    "c2lb": MagicMock(),
    "c2lb_dynamic": MagicMock(),
    "speculative_moe": MagicMock(),
    "c2lb_a3": MagicMock(),
}):
    from components.expert_load_balancing.elb.preprocessing import get_csv_dimensions, get_csv_path, AppArgs, \
            check_input


class TestGetCSVDimensions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_csv_file(self, filename, data=None, header=True):
        """create csv file"""
        file_path = os.path.join(self.temp_dir, filename)
        if data is not None:
            df = pd.DataFrame(data)
            df.to_csv(file_path, header=header, index=False)
        else:
            open(file_path, 'w').close()
        return file_path

    def test_valid_csv(self):
        test_data = np.random.rand(3, 5)
        csv_path = self._create_csv_file("valid.csv", test_data)
        with self.assertLogs(logger="msit_logger", level="INFO") as log:
            rows, cols = get_csv_dimensions(csv_path)

        self.assertEqual((rows, cols), (3, 5))
        self.assertIn(f"Successfully read", log.output[0])
        self.assertIn("Number of row=3, Number of columns=5", log.output[1])

    def test_empty_file(self):
        csv_path = self._create_csv_file("empty.csv", data=None)
        with self.assertRaises(ValueError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("文件无有效数据", str(cm.exception))

    def test_malformed_csv(self):
        csv_path = os.path.join(self.temp_dir, "malformed.csv")
        # 创建列数不一致的CSV
        with open(csv_path, 'w') as f:
            f.write("col1,col2\n1\n3,4,5")

        with self.assertRaises(ValueError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("CSV格式解析失败", str(cm.exception))

    def test_nonexistent_file(self):
        fake_path = os.path.join(self.temp_dir, "non_existent.csv")
        
        with self.assertRaises(RuntimeError) as cm:
            get_csv_dimensions(fake_path)
        self.assertIn("未知错误", str(cm.exception))

    def test_empty_dataframe(self):
        csv_path = self._create_csv_file("headers_only.csv", data=[], header=True)

        with self.assertRaises(ValueError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("文件无有效数据", str(cm.exception))
    
    @patch("pandas.read_csv")
    def test_permission_error(self, mock_read):
        """
        Exception example: Simulating file read permission error
        """
        mock_read.side_effect = PermissionError("权限拒绝")
        csv_path = os.path.join(self.temp_dir, "dummy.csv")
        open(csv_path, 'w').close()

        with self.assertRaises(RuntimeError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("未知错误", str(cm.exception))


class TestGetCSVPath(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_file(self, filename):
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "w") as f:
            f.write("mock_data")
        return file_path

    def test_valid_file_path(self):
        file_name = "decode_info.csv"
        expected_path = self._create_file(file_name)
        result_path = get_csv_path(self.temp_dir, file_name)
        self.assertEqual(result_path, expected_path)

        file_name = "prefill_info.csv"
        expected_path = self._create_file(file_name)
        result_path = get_csv_path(self.temp_dir, file_name)
        self.assertEqual(result_path, expected_path)
    
    def test_file_not_found(self):
        file_name = "non_existent.csv"
        full_path = os.path.join(self.temp_dir, file_name)
        
        with self.assertRaises(FileNotFoundError) as cm:
            get_csv_path(self.temp_dir, file_name)
        
        self.assertIn(full_path, str(cm.exception))
        self.assertIn("parameter -o", str(cm.exception))

    def test_path_concatenation(self):
        file_name = "test.csv"
        expected_path = os.path.join(self.temp_dir + os.sep, file_name)
        self._create_file(file_name)
        result_path = get_csv_path(self.temp_dir + os.sep, file_name)
        self.assertEqual(result_path, expected_path)
    
        rel_dir = "."
        file_name = "rel_test.csv"
        expected_path = os.path.join(rel_dir, file_name)
        with open(file_name, "w") as f:
            f.write("data")
        try:
            result_path = get_csv_path(rel_dir, file_name)
            self.assertEqual(result_path, expected_path)
        finally:
            os.remove(file_name)
        
    def test_invalid_filename(self):
        with self.assertRaises(FileNotFoundError):
            get_csv_path(self.temp_dir, "../invalid_path.csv")


def test_app_args_initialization():
    """测试 AppArgs 类参数是否正确初始化"""
    # 定义测试参数
    input_args = {
        "expert_popularity_csv_load_path": "/data/popularity.csv",
        "output_dir": "/output",
        "num_nodes": 4,
        "num_npus": 8,
        "share_expert_devices": 2,
        "num_redundancy_expert": 1,
        "algorithm": "dynamic_c2lb",
        "device_type": "GPU"
    }

    # 初始化对象
    args = AppArgs(**input_args)

    # 验证直接映射参数
    assert args.trace_path == input_args["expert_popularity_csv_load_path"]
    assert args.deploy_fp == input_args["output_dir"]
    assert args.n_nodes == input_args["num_nodes"]
    assert args.n_devices == input_args["num_npus"]
    assert args.n_share_expert_devices == input_args["share_expert_devices"]
    assert args.redundant_experts == input_args["num_redundancy_expert"]
    assert args.algorithm == input_args["algorithm"]
    assert args.device_type == input_args["device_type"]

    # 验证固定默认值
    assert args.selected_layers == [-1, -1]
    assert args.n_layers == 58
    assert args.max_time_in_seconds == 300
    assert args.eplb_map == "./"
    assert args.n_selected_expert == 8
    assert args.collection_interval == 16
    assert args.cpu_per_process == 12
    assert args.num_stages == 8
    assert args.enhanced is False
    assert args.black_box_annealing is False
    assert args.all2all_balance is False

    assert args.n_experts == 0 


def test_valid_input():
    """测试正常参数通过验证"""
    # 创建有效参数对象
    args = AppArgs(
        expert_popularity_csv_load_path="/data",
        output_dir="/output",
        num_nodes=4,
        num_npus=8,
        share_expert_devices=2,
        num_redundancy_expert=16,  # 16是8的倍数
        algorithm="c2lb",
        device_type="GPU"
    )
    args.selected_layers = [0, 10]  # 有效范围
    args.max_time_in_seconds = 300  # 正值
    
    # 应无异常抛出
    check_input(args)

