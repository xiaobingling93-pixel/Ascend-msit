import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd

from msserviceprofiler.modelevalstate.config.config import (
    DataStorageConfig,
    PerformanceIndex,
    BenchMarkConfig,
    OptimizerConfigField
)
from msserviceprofiler.modelevalstate.optimizer.store import DataStorage


class TestDataStorage(TestCase):
    def setUp(self):
        # 创建临时目录作为测试目录
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = DataStorageConfig(store_dir=self.test_dir)
        self.storage = DataStorage(self.config)

    def tearDown(self):
        # 清理测试目录
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_init_creates_directory(self):
        # 测试目录创建
        test_path = self.test_dir / "new_dir"
        config = DataStorageConfig(store_dir=test_path)
        storage = DataStorage(config)
        self.assertTrue(test_path.exists())
        self.assertTrue(test_path.is_dir())

    def test_load_history_position_file_not_found(self):
        # 测试加载不存在的目录
        with self.assertRaises(FileNotFoundError):
            DataStorage.load_history_position(Path("non_existent_dir"))

    def test_load_history_position_not_directory(self):
        # 测试加载文件而不是目录
        test_file = self.test_dir / "test.txt"
        test_file.touch()
        with self.assertRaises(ValueError):
            DataStorage.load_history_position(test_file)

    def test_load_history_position_empty_directory(self):
        # 测试加载空目录
        result = DataStorage.load_history_position(self.test_dir)
        self.assertIsNone(result)

    def test_load_history_position_with_data(self):
        # 创建测试数据文件
        data1 = self.test_dir / "data_storage_001.csv"
        data2 = self.test_dir / "data_storage_002.csv"
        
        # 写入测试数据
        pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_csv(data1, index=False)
        pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]}).to_csv(data2, index=False)

        # 测试数据加载
        result = DataStorage.load_history_position(self.test_dir)
        self.assertEqual(len(result), 4)  # 两个文件共4条记录

    def test_save_new_file(self):
        # 测试首次保存（创建新文件）
        perf_index = PerformanceIndex(
            qps=100.0,
            latency_avg=10.0,
            latency_p50=9.0,
            latency_p90=15.0,
            latency_p99=20.0
        )
        params = (
            OptimizerConfigField(name="param1", value="value1"),
            OptimizerConfigField(name="param2", value="value2")
        )
        bench_config = BenchMarkConfig(
            command="python bench.py --batch-size 32 --seq-len 128"
        )

        self.storage.save(perf_index, params, bench_config)
        self.assertTrue(self.storage.save_file.exists())

        # 验证文件内容
        df = pd.read_csv(self.storage.save_file)
        self.assertIn("qps", df.columns)
        self.assertIn("param1", df.columns)
        self.assertIn("batch-size", df.columns)
        self.assertIn("seq-len", df.columns)

    def test_save_append(self):
        # 测试追加保存
        perf_index = PerformanceIndex(
            qps=100.0,
            latency_avg=10.0,
            latency_p50=9.0,
            latency_p90=15.0,
            latency_p99=20.0
        )
        params = (
            OptimizerConfigField(name="param1", value="value1"),
        )
        bench_config = BenchMarkConfig(
            command="python bench.py --batch-size 32"
        )

        # 保存两次
        self.storage.save(perf_index, params, bench_config)
        self.storage.save(perf_index, params, bench_config)

        # 验证文件内容
        df = pd.read_csv(self.storage.save_file)
        self.assertEqual(len(df), 2)  # 应该有两行数据

    def test_save_with_kwargs(self):
        # 测试带额外参数的保存
        perf_index = PerformanceIndex(
            qps=100.0,
            latency_avg=10.0,
            latency_p50=9.0,
            latency_p90=15.0,
            latency_p99=20.0
        )
        params = ()
        bench_config = BenchMarkConfig(
            command="python bench.py"
        )

        self.storage.save(perf_index, params, bench_config, extra_field="extra_value")
        
        # 验证文件内容
        df = pd.read_csv(self.storage.save_file)
        self.assertIn("extra_field", df.columns)
        self.assertEqual(df["extra_field"].iloc[0], "extra_value")
