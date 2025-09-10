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

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from msserviceprofiler.modelevalstate.config.config import settings
from msserviceprofiler.modelevalstate.optimizer.benchmark import BenchMark, VllmBenchMark
from msserviceprofiler.modelevalstate.optimizer.store import DataStorage
from msserviceprofiler.msguard import GlobalConfig


class TestDataStorage:
    @classmethod
    def test_get_best_result_benchmark_is_benchmark(cls, data_storage):
        GlobalConfig.custom_return = True
        with patch('shutil.which', return_value='/path/to/benchmark'):
            data_storage.benchmark = BenchMark(settings.benchmark)
            data_storage.benchmark.benchmark_config.command.request_count = str(1000)
            data_storage.save_file = 'test.csv'
        result_df = pd.DataFrame({
            'num_prompts': [1000] * 10,
            'fitness': [9.1076, 6.3524, 7.8392, 9.3442, 6.4396, 7.8859, 6.2324, 6.3734, 6.2833, 6.3723],
            'time_to_first_token': [0.1170585, 0.4933793, 0.4758562, 3.8750339, 0.2881463, 0.1656243, 0.6007349,
                                    0.6775429, 0.3424034, 0.2853581],
            'time_per_output_token': [0.0903143, 0.0558014, 0.0383041, 0.0953723, 0.0551576, 0.043904, 0.0502942,
                                      0.0515175, 0.0522677, 0.0516389],
            'generate_speed': [2059.3861, 1989.914, 1515.2376, 2499.9194, 1954.0606, 1517.8326, 1990.0761, 1949.2964,
                               1985.6711, 1950.4967]
        })
        with patch('pandas.read_csv', return_value=result_df):
            result: pd.DataFrame = data_storage.get_best_result()
            assert (result['fitness'].values.tolist() == [9.1076, 6.3524, 9.3442, 6.2324, 6.2833])
            settings.ttft_penalty = 0
            result: pd.DataFrame = data_storage.get_best_result()
            assert (result['fitness'].values.tolist() == [9.1076, 6.3524, 9.3442, 6.2324, 6.2833])
            settings.tpot_penalty = 0
            result: pd.DataFrame = data_storage.get_best_result()
            assert (result['fitness'].values.tolist() == [9.1076, 6.3524, 9.3442, 6.2324, 6.2833])
            data_storage.benchmark = VllmBenchMark(settings.vllm_benchmark)
            data_storage.benchmark.benchmark_config.command.num_prompts = str(1000)
            settings.ttft_penalty = 3.0
            settings.tpot_penalty = 3.0
            result: pd.DataFrame = data_storage.get_best_result()
            assert (result['fitness'].values.tolist() == [6.3524, 6.2324, 6.2833, 6.3723])
        GlobalConfig.reset()

    @pytest.fixture
    def data_storage(self):
        return DataStorage(settings.data_storage, MagicMock(), MagicMock())
