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
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, MagicMock
import pandas as pd


class MockDataStorageConfig:
    def __init__(self, store_dir):
        self.store_dir = Path(store_dir)


class MockPerformanceIndex:
    def __init__(self, qps=0.0, latency_avg=0.0, latency_p50=0.0, latency_p90=0.0, latency_p99=0.0):
        self._data = {
            'qps': qps,
            'latency_avg': latency_avg,
            'latency_p50': latency_p50,
            'latency_p90': latency_p90,
            'latency_p99': latency_p99
        }
    
    def model_dump(self):
        return self._data


class MockOptimizerConfigField:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class MockBenchMarkConfig:
    def __init__(self, command):
        self.command = command

# 替换原始导入
config_mock = MagicMock()
config_mock.DataStorageConfig = MockDataStorageConfig
config_mock.PerformanceIndex = MockPerformanceIndex
config_mock.BenchMarkConfig = MockBenchMarkConfig
config_mock.OptimizerConfigField = MockOptimizerConfigField
config_mock.RUN_TIME = "test_time"

with patch.dict('sys.modules', {'msserviceprofiler.modelevalstate.config.config': config_mock}):
    from msserviceprofiler.modelevalstate.optimizer.store import DataStorage


class TestDataStorage(TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = MockDataStorageConfig(store_dir=self.test_dir)
        self.storage = DataStorage(self.config)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_init_creates_directory(self):
        test_path = self.test_dir / "new_dir"
        config = MockDataStorageConfig(store_dir=test_path)
        storage = DataStorage(config)
        self.assertTrue(test_path.exists())
        self.assertTrue(test_path.is_dir())

    def test_load_history_position_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            DataStorage.load_history_position(Path("non_existent_dir"))

    def test_load_history_position_not_directory(self):
        test_file = self.test_dir / "test.txt"
        test_file.touch()
        with self.assertRaises(ValueError):
            DataStorage.load_history_position(test_file)

    def test_load_history_position_empty_directory(self):
        result = DataStorage.load_history_position(self.test_dir)
        self.assertIsNone(result)

    def test_load_history_position_with_data(self):
        data1 = self.test_dir / "data_storage_001.csv"
        data2 = self.test_dir / "data_storage_002.csv"
        
        pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_csv(data1, index=False)
        pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]}).to_csv(data2, index=False)

        result = DataStorage.load_history_position(self.test_dir)
        self.assertEqual(len(result), 4)


    def test_load_history_with_non_csv_files(self):
        # 创建一些非CSV文件和CSV文件混合的测试数据
        data_csv = self.test_dir / "data_storage_001.csv"
        non_csv = self.test_dir / "data_storage_002.txt"
        
        pd.DataFrame({'col1': [1, 2]}).to_csv(data_csv, index=False)
        non_csv.write_text("some text")

        result = DataStorage.load_history_position(self.test_dir)
        self.assertEqual(len(result), 2)  # 应该只加载CSV文件中的数据

