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
import shutil
import re
from pathlib import Path
import unittest
import shutil
from unittest.mock import patch, MagicMock, mock_open
import csv
import pandas as pd
import pytest
from msserviceprofiler.modelevalstate.config.config import PerformanceIndex, settings, AisBenchConfig, \
    OptimizerConfigField
from msserviceprofiler.modelevalstate.optimizer.benchmark import BenchMark, parse_result, AisBench
from msserviceprofiler.msguard import GlobalConfig


class TestParseResult(unittest.TestCase):
    def test_string_with_ms(self):
        # 测试输入为字符串，且单位为ms的情况
        self.assertAlmostEqual(parse_result("123 ms"), 0.123)

    def test_string_with_us(self):
        # 测试输入为字符串，且单位为us的情况
        self.assertAlmostEqual(parse_result("456 us"), 0.000456)

    def test_string_with_other_unit(self):
        # 测试输入为字符串，但单位不是ms或us的情况
        self.assertAlmostEqual(parse_result("789 s"), 789.0)

    def test_string_without_unit(self):
        # 测试输入为字符串，但没有单位的情况
        self.assertAlmostEqual(parse_result("1010"), 1010.0)


@pytest.fixture
def benchmark():
    with patch('shutil.which', return_value='/path/to/benchmark'):
        benchmark = BenchMark(settings.benchmark)
        benchmark.throughput_type = "common"
        benchmark.benchmark_config.output_path = Path("./result")
    return benchmark


@pytest.fixture
def results_per_request_file(tmpdir):
    file_path = Path(tmpdir).joinpath("results_per_request_202507181613.json")
    data = {
        "1": {
            "input_len": 1735,
            "output_len": 1,
            "prefill_bsz": 4,
            "decode_bsz": [],
            "req_latency": 13058.372889645398,
            "latency": [
                13058.23168065399
            ],
            "queue_latency": [
                12598012
            ],
            "input_data": "", "output": ""
        },
        "2": {
            "prefill_bsz": 4,
            "decode_bsz": [],
            "req_latency": 15173.639830201864,
            "latency": [
                15173.517209477723
            ],
            "queue_latency": [
                14708480
            ],
            "input_data": "", "output": ""
        },
        "3": {
            "input_len": 1777,
            "output_len": 3,
            "prefill_bsz": 4,
            "decode_bsz": [
                157,
                157
            ],
            "req_latency": 15456.984990276396,
            "latency": [
                15178.787489421666,
                208.4683496505022,
                69.54238004982471
            ],
            "queue_latency": [
                14711475,
                127888,
                3709
            ],
            "input_data": "", "output": "\t\tif ("
        },
        "4": {
            "input_len": 1770,
            "output_len": 3,
            "prefill_bsz": 4,
            "decode_bsz": [
                157,
                157
            ],
            "req_latency": 15481.421849690378,
            "latency": [
                14745.695400051773,
                670.0493693351746,
                64.66158013790846
            ],
            "queue_latency": [
                14280800,
                584221,
                3686
            ],
            "input_data": "", "output": "Passage "
        },
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path


class TestBenchMark:
    @classmethod
    def test_get_req_token_info(cls, benchmark, results_per_request_file):
        benchmark.benchmark_config.command.save_path = str(Path(results_per_request_file).parent)
        result = benchmark.get_req_token_info()
        expect_result = {
            "http_rid": ["1", "3", "4"],
            "recv_token_size": [1735, 1777, 1770],
            "reply_token_size": [1, 3, 3]
        }
        assert result == expect_result
        assert benchmark.get_req_token_info(results_per_request_file) == expect_result

    @patch("pathlib.Path.iterdir")
    def test_get_performance_index_no_result_perf(self, mock_iterdir, benchmark):
        GlobalConfig.custom_return = True
        mock_file_common = MagicMock()
        mock_file_common.name = "result_common_20250517031241120.csv"
        mock_iterdir.return_value = [mock_file_common, ]
        df_common = pd.DataFrame({
            "OutputGenerateSpeed": ["1995.7208 token/s"],
            "Returned": ["3000( 100.0% )"],
        })
        with patch("pandas.read_csv", side_effect=[df_common, ]):
            with pytest.raises(ValueError, match="Not Found first_token_time."):
                benchmark.get_performance_index()
        GlobalConfig.reset()

    @patch("pathlib.Path.iterdir")
    def test_get_performance_index_success(self, mock_iterdir, benchmark):
        GlobalConfig.custom_return = True
        mock_file_common = MagicMock()
        mock_file_common.name = "result_common_20250517031241120.csv"
        mock_file_perf = MagicMock()
        mock_file_perf.name = "result_perf_20250517041302288.csv"
        mock_iterdir.return_value = [mock_file_common, mock_file_perf]

        df_common = pd.DataFrame({
            "OutputGenerateSpeed": ["1995.7208 token/s"],
            "Returned": ["3000( 100.0% )"],
        })
        df_perf = pd.DataFrame({
            "FirstTokenTime": ["572.8072 ms", "572.8072 ms", "572.8072 ms", "572.8072 ms", "572.8072 ms", "572.8072 ms",
                               "572.8072 ms"],
            "GeneratedTokenSpeed": ["7.5371 token/s", "7.5371 token/s", "7.5371 token/s", "7.5371 token/s",
                                    "7.5371 token/s", "7.5371 token/s", "7.5371 token/s"],
            "DecodeTime": ["127.9866 ms", "127.9866 ms", "127.9866 ms", "127.9866 ms", "127.9866 ms", "127.9866 ms",
                           "127.9866 ms"],
        })

        with patch("pandas.read_csv", side_effect=[df_common, *[df_perf] * 23]):
            result = benchmark.get_performance_index()
            assert isinstance(result, PerformanceIndex)
            assert result.generate_speed == 1995.7208
            assert result.time_to_first_token == 0.5728072
            assert result.time_per_output_token == 0.1279866
        GlobalConfig.reset()

    @patch("pathlib.Path.iterdir")
    def test_get_performance_with_custom_algorithm(self, mock_iterdir, benchmark):
        GlobalConfig.custom_return = True
        mock_file_common = MagicMock()
        mock_file_common.name = "result_common_20250517031241120.csv"
        mock_file_perf = MagicMock()
        mock_file_perf.name = "result_perf_20250517041302288.csv"
        mock_iterdir.return_value = [mock_file_common, mock_file_perf]
        benchmark.benchmark_config.performance_config.time_per_output_token.algorithm = "max"
        benchmark.benchmark_config.performance_config.time_to_first_token.algorithm = "max"
        df_common = pd.DataFrame({
            "OutputGenerateSpeed": ["1998.7208 token/s"],
            "Returned": ["3000( 100.0% )"],
        })
        df_perf = pd.DataFrame({
            "FirstTokenTime": ["572.8072 ms", "633.8072 ms", "633.8072 ms", "633.8072 ms", "633.8072 ms", "633.8072 ms",
                               "633.8072 ms"],
            "GeneratedTokenSpeed": ["7.5371 token/s", "8.5371 token/s", "8.5371 token/s", "8.5371 token/s",
                                    "8.5371 token/s", "8.5371 token/s", "8.5371 token/s"],
            "DecodeTime": ["127.9866 ms", "144.9866 ms", "144.9866 ms", "144.9866 ms", "144.9866 ms", "144.9866 ms",
                           "144.9866 ms"],
        })

        with patch("pandas.read_csv", side_effect=[df_common, *[df_perf] * 23]):
            result = benchmark.get_performance_index()
            assert isinstance(result, PerformanceIndex)
            assert result.generate_speed == 1998.7208
            assert result.time_to_first_token == 0.6338072
            assert result.time_per_output_token == float("144.9866") / 10 ** 3
        GlobalConfig.reset()


class TestAisbench(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test")
        self.test_dir.mkdir(exist_ok=True)
        self.aisbench_dir = self.test_dir / "outputs"
        self.aisbench_dir.mkdir(exist_ok=True)
        self.performance_dir = self.aisbench_dir / "performances"
        self.performance_dir.mkdir(exist_ok=True)
        self.output_dir = self.performance_dir / "api_file"
        self.output_dir.mkdir(exist_ok=True)
        self.csv_path = self.output_dir / "gsm8kdataset.csv"
        self.json_path = self.output_dir / "gsm8kdataset.json"
        json_data = {
            "Total Requests": {
                "total": 84
            },
            "Success Requests": {
                "total": 84
            },
            "Request Throughput": {
                "total": "2.4221 req/s"
            },
            "Output Token Throughput": {
                "total": "1240.1267 token/s"
            }
        }
        with open(self.json_path, 'w') as f:
            json.dump(json_data, f)
        data = [
            {
                "Performance Parameters": "TTFT",
                "Stage": "total",
                "Average": "146.1383 ms",
            },
            {
                "Performance Parameters": "TPOT",
                "Stage": "total",
                "Average": "30.2947 ms",
            }
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as csvfile:
            # 获取所有列名
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    @patch("msserviceprofiler.modelevalstate.config.custom_command.shutil.which")
    def test_get_performance_metric(self, mock_which):
        mock_which.return_value = "/usr/local/bin/aisbench"
        self.config = AisBenchConfig()
        self.config.output_path = self.test_dir
        assert AisBench(self.config).get_performance_metric('ttft') == 0.1461383

    @patch("msserviceprofiler.modelevalstate.config.custom_command.shutil.which")
    def test_get_performance_index(self, mock_which):
        mock_which.return_value = "/usr/local/bin/aisbench"
        self.config = AisBenchConfig()
        self.config.output_path = self.test_dir
        assert AisBench(self.config).get_performance_index().generate_speed == 1240.1267


class TestBeforeRun(unittest.TestCase):
    def setUp(self):
        # 创建临时测试环境
        self.test_dir = Path("ais_bench")
        self.test_dir.mkdir(exist_ok=True)
        self.benchmark_dir = self.test_dir / "benchmark"
        self.benchmark_dir.mkdir(exist_ok=True)
        self.config_dir = self.benchmark_dir / "configs"
        self.config_dir.mkdir(exist_ok=True)
        self.model_dir = self.config_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        code_content = '''from ais_bench.benchmark.models import VLLMCustomAPIChatStream
        models = [
            dict(
                attr="service",
                type=VLLMCustomAPIChatStream,
                abbr='vllm-api-stream-chat',
                path="/data/models/llama3-8b",
                model="llama3-8b",
                request_rate=36,
                retry=2,
                host_ip="127.0.0.1",
                host_port=31015,
                max_out_len=512,
                batch_size=1000,
                trust_remote_code=False,
                generation_kwargs=dict(
                    temperature=0.5,
                    top_k=10,
                    top_p=0.95,
                    seed=None,
                    repetition_penalty=1.03,
                )
        )
        ]'''
        target_file = self.model_dir / "api.py"
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(code_content)

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    @patch("msserviceprofiler.modelevalstate.config.custom_command.shutil.which")
    @patch('importlib.import_module')
    def test_before_run_file_exists(self, mock_import, mock_which):
        """
        测试文件存在且成功修改 request_rate 和 batch_size 的情况。
        """
        # 模拟导入模块
        mock_which.return_value = "/usr/local/bin/aisbench"
        self.config = AisBenchConfig()
        self.config.command.models = "api"
        mock_module = MagicMock()
        mock_module.__file__ = 'ais_bench/__init__.py'
        mock_import.return_value = mock_module
        

        # 模拟运行参数
        support_field = [
            OptimizerConfigField(name="CONCURRENCY", 
                               config_position="env", 
                               min=25, max=300, dtype="int", value=100),
            OptimizerConfigField(name="REQUESTRATE",
                               config_position="env", 
                               min=1, max=25, dtype="int", value=100)
        ]
        # 调用方法
        GlobalConfig.custom_return = True
        AisBench(self.config).before_run(support_field)
        pattern = re.compile(r"request_rate\s*=\s*(\d+)")
        with open('ais_bench/benchmark/configs/models/api.py', 'r', encoding='utf-8') as f:
            content = f.read()
            match = pattern.search(content)
            assert int(match.group(1)) == 100
        GlobalConfig.reset()
