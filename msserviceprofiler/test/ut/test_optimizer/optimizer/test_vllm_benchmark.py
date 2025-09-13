# -*- coding: utf-8 -*-
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
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from msserviceprofiler.modelevalstate.config.config import PerformanceIndex, settings
from msserviceprofiler.modelevalstate.optimizer.benchmark import VllmBenchMark
from msserviceprofiler.msguard import GlobalConfig


class TestVllmBenchMark:
    # 测试数据
    TEST_DATA = [{
        "output_throughput": 200,
        "mean_ttft_ms": 2000,
        "mean_tpot_ms": 1000,
        "num_prompts": 15,
        "completed": 15
    },
        {
            "output_throughput": 200,
            "p99_ttft_ms": 2000,
            "p99_tpot_ms": 1000,
            "num_prompts": 15,
            "completed": 15
        }
    ]

    # 模拟文件系统
    @patch('msserviceprofiler.msguard.security.io.walk_s')
    @patch('msserviceprofiler.modelevalstate.optimizer.benchmark.open', new_callable=mock_open, 
        read_data=json.dumps(TEST_DATA[1]))
    @patch('os.scandir')
    def test_get_performance_index(self, mock_scandir, mock_iterdir, tmpdir):
        GlobalConfig.custom_return = True
        # 模拟 iterdir 返回的文件列表
        mock_iterdir.return_value = [Path('test1.json'), Path('test2.json')]
        mock_scandir.return_value = []

        # 创建 VllmBenchMark 实例
        benchmark = VllmBenchMark(settings.vllm_benchmark)
        benchmark.benchmark_config = MagicMock()
        benchmark.benchmark_config.output_path = Path(tmpdir)
        benchmark.benchmark_config.performance_config.time_to_first_token.metric = 'p99_ttft_ms'
        benchmark.benchmark_config.performance_config.time_per_output_token.metric = 'p99_tpot_ms'

        # 调用 get_performance_index 方法
        result = benchmark.get_performance_index()

        # 验证结果
        assert isinstance(result, PerformanceIndex)
        GlobalConfig.reset()

    # 测试没有.json文件的情况
    @patch('msserviceprofiler.msguard.security.io.walk_s')
    def test_get_performance_index_no_json(self, mock_iterdir):
        # 模拟 iterdir 返回的文件列表
        GlobalConfig.custom_return = True
        mock_iterdir.return_value = [Path('test1.txt'), Path('test2.csv')]

        # 创建 VllmBenchMark 实例
        benchmark = VllmBenchMark(settings.vllm_benchmark)

        # 调用 get_performance_index 方法
        result = benchmark.get_performance_index()

        # 验证结果
        assert isinstance(result, PerformanceIndex)
        assert result.generate_speed is None
        assert result.time_to_first_token is None
        assert result.time_per_output_token is None
        assert result.success_rate is None
        GlobalConfig.reset()

    @patch('msserviceprofiler.msguard.security.io.walk_s')
    @patch('msserviceprofiler.modelevalstate.optimizer.benchmark.open_s', new_callable=mock_open, 
        read_data=json.dumps(TEST_DATA[0]))
    @patch('os.scandir')
    def test_get_performance_index_mean(self, mock_scandir, mock_iterdir, tmpdir):
        # 模拟 iterdir 返回的文件列表
        GlobalConfig.custom_return = True
        mock_iterdir.return_value = [Path('test1.json'), Path('test2.json')]
        mock_scandir.return_value = []

        # 创建 VllmBenchMark 实例
        benchmark = VllmBenchMark(settings.vllm_benchmark)
        benchmark.benchmark_config = MagicMock()
        benchmark.benchmark_config.output_path = Path(tmpdir)
        benchmark.benchmark_config.performance_config.time_to_first_token.metric = 'mean_ttft_ms'
        benchmark.benchmark_config.performance_config.time_per_output_token.metric = 'mean_tpot_ms'

        # 调用 get_performance_index 方法
        result = benchmark.get_performance_index()

        # 验证结果
        assert isinstance(result, PerformanceIndex)
        GlobalConfig.reset()
