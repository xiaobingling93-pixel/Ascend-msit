# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
import tempfile
from unittest.mock import patch

from msprechecker.prechecker.utils import logger
from msprechecker.prechecker.hardware_capacity.time_analyze import TimeAnalyze


class TestTimeAnalyze(unittest.TestCase):
    def setUp(self):
        self.test_run_time = {
            "task1": 10.5,
            "task2": 10.6,
            "task3": 10.4,
            "task4": 11.0,
            "task5": 10.7
        }
        self.extreme_run_time = {
            "task1": 10.0,
            "task2": 100.0,
            "task3": 10.1
        }
        self.empty_run_time = {}
        self.zero_run_time = {"task1": 0.0, "task2": 0.0}

    def test_time_analyze_normal_case(self):
        analyzer = TimeAnalyze(self.test_run_time)
        result = analyzer.time_analyze()

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "task4")
        self.assertAlmostEqual(result[1], 11.0)
        self.assertLess(result[2], TimeAnalyze.RATIO_THRESHOLD)
        self.assertFalse(result[3])

    def test_time_analyze_extreme_case(self):
        analyzer = TimeAnalyze(self.extreme_run_time)
        result = analyzer.time_analyze()

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "task2")
        self.assertAlmostEqual(result[1], 100.0)
        self.assertGreater(result[2], TimeAnalyze.RATIO_THRESHOLD)
        self.assertTrue(result[3])

    def test_time_analyze_empty_input(self):
        analyzer = TimeAnalyze(self.empty_run_time)
        self.assertEqual(analyzer.time_analyze(), ())

    def test_time_analyze_zero_values(self):
        analyzer = TimeAnalyze(self.zero_run_time)
        with self.assertRaises(RuntimeError):
            analyzer.time_analyze()

    def test_time_analyze_with_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_data.txt")
            with open(temp_file, 'w') as f:
                f.write("test data")

            # This part is just to demonstrate temp file usage
            # In real case, you might want to read data from this file
            analyzer = TimeAnalyze(self.test_run_time)
            result = analyzer.time_analyze()

            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(temp_file))

        self.assertFalse(os.path.exists(temp_file))

    @patch.object(logger, 'error')
    def test_time_analyze_logging(self, mock_logger):
        analyzer = TimeAnalyze({})
        self.assertEqual(analyzer.time_analyze(), ())
        mock_logger.assert_called_once_with("Running time is undefined.")


class DerivedTimeAnalyze(TimeAnalyze):
    def __init__(self, run_time):
        super().__init__(run_time)


class TestDerivedTimeAnalyze(unittest.TestCase):
    def test_derived_class_usage(self):
        test_data = {"task1": 10.0, "task2": 10.1}
        analyzer = DerivedTimeAnalyze(test_data)
        result = analyzer.time_analyze()
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "task2")
        self.assertAlmostEqual(result[1], 10.1)
        self.assertLess(result[2], TimeAnalyze.RATIO_THRESHOLD)
        self.assertFalse(result[3])
