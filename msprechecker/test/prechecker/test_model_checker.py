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
import tempfile
import unittest
import hashlib
from unittest.mock import patch

from msprechecker.prechecker.register import CheckResult
from msprechecker.prechecker.model_checker import (
    ModelSizeChecker,
    ModelSha256Collecter,
    get_file_sizes,
    get_file_sha256s,
    DEEPSEEK_R1_FP8_WEIGHT_SIZE,
)


class TestModelChecker(unittest.TestCase):
    def test_get_file_sizes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "wb") as f:
                f.write(b"test content")
            
            result = get_file_sizes(os.path.join(temp_dir, "*.txt"))
            self.assertIsNotNone(result)
            self.assertIn("test.txt", result)
            self.assertEqual(result["test.txt"]["size"], 12)

    def test_get_file_sha256s_empty_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "empty.txt")
            with open(test_file, "wb"):
                pass
            
            result = get_file_sha256s(os.path.join(temp_dir, "*.txt"))
            self.assertIsNotNone(result)
            self.assertIn("empty.txt", result)
            expected_hash = hashlib.sha256().hexdigest()
            self.assertEqual(result["empty.txt"]["sha256sum"], expected_hash)

    def test_get_file_sha256s_small_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "small.txt")
            with open(test_file, "wb") as f:
                f.write(b"test content")
            
            result = get_file_sha256s(os.path.join(temp_dir, "*.txt"))
            self.assertIsNotNone(result)
            self.assertIn("small.txt", result)
            expected_hash = hashlib.sha256(b"test content").hexdigest()
            self.assertEqual(result["small.txt"]["sha256sum"], expected_hash)

    def test_get_file_sha256s_large_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "large.txt")
            content = b"a" * 5000  # Larger than default block size
            with open(test_file, "wb") as f:
                f.write(content)
            
            result = get_file_sha256s(os.path.join(temp_dir, "*.txt"), block_size=1000, num_blocks=5)
            self.assertIsNotNone(result)
            self.assertIn("large.txt", result)
            expected_hash = hashlib.sha256(content).hexdigest()
            self.assertEqual(result["large.txt"]["sha256sum"], expected_hash)


class TestModelSizeChecker(unittest.TestCase):
    def setUp(self):
        self.checker = ModelSizeChecker()

    def test_to_g_size(self):
        result = self.checker.to_g_size(1073741824)  # 1GB
        self.assertEqual(result, "1.00G")

    @patch('msprechecker.prechecker.model_checker.get_model_path_from_mindie_config')
    def test_collect_env_no_model(self, mock_get_model):
        mock_get_model.return_value = (None, None)
        result = self.checker.collect_env()
        self.assertIsNone(result)

    @patch('msprechecker.prechecker.model_checker.get_model_path_from_mindie_config')
    @patch('msprechecker.prechecker.model_checker.get_file_sizes')
    def test_collect_env_with_model(self, mock_get_sizes, mock_get_model):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_model.return_value = ("test_model", temp_dir)
            mock_get_sizes.side_effect = [{"config.json": {"size": 100}}, {"model.safetensors": {"size": 1000}}]
            
            result = self.checker.collect_env()
            self.assertIsNotNone(result)
            self.assertEqual(result["model_name"], "test_model")
            self.assertIn("model_json_size", result)
            self.assertIn("model_weight_size", result)

    @patch('msprechecker.prechecker.model_checker.is_deepseek_model')
    @patch('msprechecker.prechecker.model_checker.show_check_result')
    def test_do_precheck_not_deepseek(self, mock_show_result, mock_is_deepseek):
        mock_is_deepseek.return_value = False
        model_config = {"model_name": "other_model", "model_weight_size": {"a": {"size": 100}}}
        self.checker.do_precheck(model_config)
        mock_show_result.assert_not_called()

    @patch('msprechecker.prechecker.model_checker.is_deepseek_model')
    @patch('msprechecker.prechecker.model_checker.show_check_result')
    def test_do_precheck_valid_fp8_size(self, mock_show_result, mock_is_deepseek):
        mock_is_deepseek.return_value = True
        model_config = {
            "model_name": "deepseek",
            "model_weight_size": {"a": {"size": DEEPSEEK_R1_FP8_WEIGHT_SIZE}}
        }
        self.checker.do_precheck(model_config)
        self.assertEqual(mock_show_result.call_count, 1)
        args, _ = mock_show_result.call_args
        self.assertEqual(args[2], CheckResult.OK)

    @patch('msprechecker.prechecker.model_checker.is_deepseek_model')
    @patch('msprechecker.prechecker.model_checker.show_check_result')
    def test_do_precheck_invalid_size(self, mock_show_result, mock_is_deepseek):
        mock_is_deepseek.return_value = True
        model_config = {
            "model_name": "deepseek",
            "model_weight_size": {"a": {"size": 100}}  # Much smaller than expected
        }
        self.checker.do_precheck(model_config)
        self.assertEqual(mock_show_result.call_count, 1)
        args, _ = mock_show_result.call_args
        self.assertEqual(args[2], CheckResult.ERROR)


class TestModelSha256Collecter(unittest.TestCase):
    def setUp(self):
        self.collecter = ModelSha256Collecter()

    @patch('msprechecker.prechecker.model_checker.get_model_path_from_mindie_config')
    def test_collect_env_no_model(self, mock_get_model):
        mock_get_model.return_value = (None, None)
        result = self.collecter.collect_env()
        self.assertIsNone(result)

    @patch('msprechecker.prechecker.model_checker.get_model_path_from_mindie_config')
    @patch('msprechecker.prechecker.model_checker.get_file_sha256s')
    def test_collect_env_with_model(self, mock_get_sha256s, mock_get_model):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_model.return_value = ("test_model", temp_dir)
            mock_get_sha256s.side_effect = [
                {"config.json": {"sha256sum": "abc123"}},
                {"model.safetensors": {"sha256sum": "def456"}}
            ]
            
            result = self.collecter.collect_env()
            self.assertIsNotNone(result)
            self.assertEqual(result["model_name"], "test_model")
            self.assertIn("model_json_sha256", result)
            self.assertIn("model_weight_sha256", result)

    @patch('msprechecker.prechecker.model_checker.logger')
    def test_do_precheck(self, mock_logger):
        model_config = {"model_name": "test", "model_weight_sha256": {"a": "b"}}
        self.collecter.do_precheck(model_config)
        mock_logger.warning.assert_called_once()