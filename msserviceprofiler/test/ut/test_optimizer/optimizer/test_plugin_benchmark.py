# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from msserviceprofiler.modelevalstate.config.config import PerformanceIndex, get_settings
from msserviceprofiler.modelevalstate.optimizer.plugins.benchmark import VllmBenchMark
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
    @patch('msserviceprofiler.msguard.security.io.open_s', new_callable=mock_open, 
        read_data=json.dumps(TEST_DATA[1]))
    @patch('os.scandir')
    def test_get_performance_index(self, mock_scandir, mock_iterdir, tmpdir):
        GlobalConfig.custom_return = True
        # 模拟 iterdir 返回的文件列表
        mock_iterdir.return_value = [Path('test1.json'), Path('test2.json')]
        mock_scandir.return_value = []

        # 创建 VllmBenchMark 实例
        benchmark = VllmBenchMark(get_settings().vllm_benchmark)
        benchmark.benchmark_config = MagicMock()
        benchmark.benchmark_config.output_path = Path(tmpdir)
        benchmark.benchmark_config.performance_config.time_to_first_token.metric = 'p99_ttft_ms'
        benchmark.benchmark_config.performance_config.time_per_output_token.metric = 'p99_tpot_ms'

        # 调用 get_performance_index 方法
        result = benchmark.get_performance_index()

        # 验证结果
        assert isinstance(result, PerformanceIndex)
        GlobalConfig.reset()

    @patch('msserviceprofiler.msguard.security.io.walk_s')
    @patch('msserviceprofiler.msguard.security.io.open_s', new_callable=mock_open, 
        read_data=json.dumps(TEST_DATA[0]))
    @patch('os.scandir')
    def test_get_performance_index_mean(self, mock_scandir, mock_iterdir, tmpdir):
        # 模拟 iterdir 返回的文件列表
        GlobalConfig.custom_return = True
        mock_iterdir.return_value = [Path('test1.json'), Path('test2.json')]
        mock_scandir.return_value = []

        # 创建 VllmBenchMark 实例
        benchmark = VllmBenchMark(get_settings().vllm_benchmark)
        benchmark.benchmark_config = MagicMock()
        benchmark.benchmark_config.output_path = Path(tmpdir)
        benchmark.benchmark_config.performance_config.time_to_first_token.metric = 'mean_ttft_ms'
        benchmark.benchmark_config.performance_config.time_per_output_token.metric = 'mean_tpot_ms'

        # 调用 get_performance_index 方法
        result = benchmark.get_performance_index()

        # 验证结果
        assert isinstance(result, PerformanceIndex)
        GlobalConfig.reset()
