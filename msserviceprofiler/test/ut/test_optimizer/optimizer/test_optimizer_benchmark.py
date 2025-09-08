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
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from msserviceprofiler.modelevalstate.config.config import PerformanceIndex, settings
from msserviceprofiler.modelevalstate.optimizer.benchmark import BenchMark
from msserviceprofiler.msguard import GlobalConfig


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

