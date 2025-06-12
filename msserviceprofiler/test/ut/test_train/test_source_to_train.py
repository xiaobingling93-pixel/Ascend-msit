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

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import os

import json
import shutil
import sqlite3
import pandas as pd
import numpy as np

from msserviceprofiler.modelevalstate.train.source_to_train import (
    DatabaseConnector,
    source_to_model,
    req_decodetimes,
    read_batch_exec_data
)


class TestSourceToTrain(unittest.TestCase):
    """测试数据预处理和训练流程的功能"""

    def setUp(self):
        # 创建临时测试环境
        self.test_dir = Path("test_source_to_train")
        self.test_dir.mkdir(exist_ok=True)

        # 创建样本数据库
        self.db_path = self.test_dir / "profiler.db"
        self.create_sample_database()

        # 创建样本CSV文件
        self.create_sample_csv()

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    def create_sample_database(self):
        """创建样本SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建batch表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch (
            name TEXT,
            res_list TEXT,
            start_time REAL,
            end_time REAL,
            batch_size REAL,
            batch_type TEXT,
            during_time REAL,
            dp0_rid TEXT,
            dp0_size REAL,
            dp0_forward REAL
        )
        """)

        # 插入样本数据
        batch_data = [
            ("batchFrameworkProcessing", "[{'rid': 101, 'iter': 0}]", 1749451414153,
             1749451414154, 1, "Prefill", 0.22175, "['0']", 1, 68.71325),
            ("batchFrameworkProcessing", "[{'rid': 102, 'iter': 0}]", 1749451414154,
             1749451414155, 1, "Prefill", 0.223, "['0']", 1, 69.2),
            ("modelExec", "[{'rid': 101, 'iter': 0}]", 1749451414158,
             1749451414167, 1, "Prefill", 100, "['0']", 1, 50),
            ("modelExec", "[{'rid': 103, 'iter': 1}]", 1749451414170,
             1749451414190, 1, "Decode", 30, "['0']", 1, 20),
            ("batchFrameworkProcessing", "[{'rid': 104, 'iter': 0}]", 1111,
             1749451414154, 1, "Prefill", 0.22175, "['0']", 1, 68.71325),
            ("batchFrameworkProcessing", "[{'rid': 102, 'iter': 0}]", 22,
             1749451414155, 1, "Prefill", 0.223, "['0']", 1, 69.2),
            ("modelExec", "[{'rid': 101, 'iter': 0}]", 45,
             1749451414167, 1, "Prefill", 100, "['0']", 1, 50),
            ("modelExec", "[{'rid': 103, 'iter': 1}]", 4867,
             1749451414190, 1, "Decode", 30, "['0']", 1, 20)
        ]
        cursor.executemany(
            "INSERT INTO batch (name,res_list, start_time,end_time,batch_size,batch_type,"
            "during_time,dp0_rid,dp0_size,dp0_forward) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch_data)

        # 创建batch_exec表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_exec (
            batch_id INTEGER PRIMARY KEY,
            name TEXT,
            pid INTEGER,
            start_time REAL,
            end_time REAL
        )
        """)

        # 插入样本数据
        exec_data = [
            (1, 'forward', 1001, 1000.0, 1500.0),
            (2, 'forward', 1002, 2000.0, 2500.0)
        ]
        cursor.executemany("INSERT INTO batch_exec (batch_id, name, pid, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
                           exec_data)

        # 创建batch_req表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_req (
            batch_id INTEGER,
            req_id TEXT,
            rid TEXT,
            iter INTEGER,
            block INTEGER
        )
        """)

        # 插入样本数据
        req_data = [
            (0, "101", "101", "3", 128),
            (1, "102", "102", "3", 256),
            (2, "103", "103", "5", 192)
        ]
        cursor.executemany("INSERT INTO batch_req (batch_id, req_id, rid, iter, block) VALUES (?, ?, ?, ?, ?)",
                           req_data)

        conn.commit()
        conn.close()

    def create_sample_csv(self):
        """创建样本CSV文件"""
        # 创建request.csv
        request_data = pd.DataFrame({
            "http_rid": ["101", "102", "103"],
            "start_time": ["1", "2", "3"],
            "recv_token_size": ["256", "512", "384"],
            "reply_token_size": ["128", "256", "192"],
            "execution_time": ["1", "1", "2"],
            "queue_wait_time": ["1", "1", "2"],
            "first_token_latency": ["1", "2", "3"]
        })
        request_data.to_csv(self.test_dir / "request.csv", index=False)

        # 创建kvcache.csv (仅用于vllm模式)
        kvcache_data = pd.DataFrame({
            "domain": ["KVCache", "KVCache", "KVCache"],
            "rid": ["101", "102", "103"],
            "timestamp": ["1749451415160", "1749451415161", "1749451415162"],
            "name": ["blocks", "blocks", "blocks"],
            "device_kvcache_left": ["128", "256", "192"]
        })
        kvcache_data.to_csv(self.test_dir / "kvcache.csv", index=False)

    def test_database_connector(self):
        """测试数据库连接器"""
        # 测试正常连接
        db_conn = DatabaseConnector(str(self.db_path))
        cursor = db_conn.connect()
        self.assertIsNotNone(cursor)

        # 测试读取batch_exec数据
        exec_rows = read_batch_exec_data(cursor)
        self.assertEqual(len(exec_rows), 2)
        self.assertEqual(exec_rows[0][1], "forward")

        # 测试关闭连接
        db_conn.close()

    @patch("msserviceprofiler.modelevalstate.train.source_to_train.pretrain")
    def test_source_to_model_mindie(self, mock_pretrain):
        """测试Mindie数据预处理流程"""
        # 执行数据处理
        source_to_model(self.test_dir, model_type=None)

        # 验证输出目录创建
        output_csv = self.test_dir / "output_csv"
        self.assertTrue(output_csv.exists())

        # 验证输出文件存在
        for pid_dir in output_csv.iterdir():
            if pid_dir.is_dir():
                self.assertTrue((pid_dir / "feature.csv").exists())

    @patch("msserviceprofiler.modelevalstate.train.source_to_train.pretrain")
    @patch("msserviceprofiler.modelevalstate.train.source_to_train.process_execution_data_vllm")
    def test_source_to_model_vllm(self, patch_pretrain, patch_process_execution_data_vllm):
        """测试vLLM数据预处理流程"""
        # 执行数据处理
        source_to_model(self.test_dir, model_type="vllm")

        # 验证输出目录创建
        output_csv = self.test_dir / "output_csv"
        self.assertTrue(output_csv.exists())

        # 验证输出文件存在
        for pid_dir in output_csv.iterdir():
            if pid_dir.is_dir():
                self.assertTrue((pid_dir / "feature.csv").exists())

    def test_req_decodetimes(self):
        """测试解码时间处理"""
        # 执行处理
        json_file = self.test_dir / "output" / "req_id_and_decode_num.json"
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        # 创建空 JSON 文件
        with open(json_file, "w", encoding="utf-8") as f:
            f.write("")
        req_decodetimes(self.test_dir, self.test_dir / "output")

        # 验证JSON文件生成
        self.assertTrue(json_file.exists())

        # 验证JSON内容
        with open(json_file, "r") as f:
            data = json.load(f)
            self.assertEqual(len(data), 3)
            self.assertEqual(data["0"], 128)


if __name__ == "__main__":
    unittest.main()
