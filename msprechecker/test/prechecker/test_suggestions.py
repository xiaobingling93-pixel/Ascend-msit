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
from unittest.mock import patch

from msprechecker.prechecker.suggestions import (
    is_condition_met,
    convert_value_type,
    is_value_met_suggestions,
    suggestion_rule_checker,
    DOMAIN,
    CONFIG,
)
from msprechecker.prechecker.register import CheckResult


class TestConditionChecker(unittest.TestCase):
    def test_is_condition_met(self):
        env_info = {"version": "2.1", "os": "linux"}
        condition = {"version": [">2.0", "2.1"], "os": ["linux"]}
        self.assertTrue(is_condition_met(env_info, condition))

    def test_is_condition_not_met(self):
        env_info = {"version": "1.9", "os": "linux"}
        condition = {"version": [">2.0", "2.1"], "os": ["linux"]}
        self.assertFalse(is_condition_met(env_info, condition))

    def test_is_condition_met_with_missing_key(self):
        env_info = {"os": "linux"}
        condition = {"version": [">2.0"], "os": ["linux"]}
        self.assertFalse(is_condition_met(env_info, condition))


class TestValueConversion(unittest.TestCase):
    def test_convert_value_type_for_env_vars(self):
        result = convert_value_type(123, DOMAIN.environment_variables)
        self.assertEqual(result, "123")

    def test_convert_value_type_for_other_domains(self):
        result = convert_value_type(123, DOMAIN.mindie_config)
        self.assertEqual(result, 123)

    def test_convert_none_value(self):
        result = convert_value_type(None, DOMAIN.environment_variables)
        self.assertIsNone(result)


class TestValueSuggestionChecker(unittest.TestCase):
    def setUp(self):
        self.current_configs = {"feature": "experimental"}

    @patch("msprechecker.prechecker.match_special_value.is_value_met_special_suggestions")
    def test_is_value_met_suggestions_with_normal_values(self, mock_special):
        mock_special.return_value = False
        current_value = "stable"
        suggested_values = ["stable", "beta"]
        result = is_value_met_suggestions(current_value, suggested_values, self.current_configs)
        self.assertTrue(result)

    def test_is_value_met_suggestions_with_empty_suggestions(self):
        current_value = "some_value"
        suggested_values = []
        is_value_met_suggestions(current_value, suggested_values, self.current_configs)
        self.assertIsNotNone(current_value)

    @patch("msprechecker.prechecker.match_special_value.is_value_met_special_suggestions")
    def test_is_value_met_suggestions_with_special_values(self, mock_special):
        mock_special.return_value = True
        current_value = "some_value"
        suggested_values = ["=special_condition"]
        result = is_value_met_suggestions(current_value, suggested_values, self.current_configs)
        self.assertTrue(result)


class TestSuggestionRuleChecker(unittest.TestCase):
    def setUp(self):
        self.env_info = {"version": "2.1"}
        self.domain = DOMAIN.environment_variables
        self.simple_rule = {
            CONFIG.name: "HOST",
            CONFIG.value: "localhost",
            CONFIG.reason: "配置为本机地址"
        }
        self.complex_rule = {
            CONFIG.name: "PORT",
            CONFIG.suggestions: [
                {
                    CONFIG.value: "8080",
                    CONFIG.suggested: {
                        CONFIG.condition: {"version": ["2.1"]},
                        CONFIG.reason: "推荐端口"
                    }
                }
            ]
        }

    def test_simple_rule_check(self):
        current_configs = {"HOST": "127.0.0.1"}
        result = suggestion_rule_checker(current_configs, self.simple_rule, self.env_info, self.domain)
        self.assertEqual(result[0], CheckResult.ERROR)

    def test_complex_rule_with_met_condition(self):
        current_configs = {"PORT": "8081"}
        result = suggestion_rule_checker(current_configs, self.complex_rule, self.env_info, self.domain)
        self.assertEqual(result[0], CheckResult.ERROR)

    def test_complex_rule_with_unmet_condition(self):
        env_info = {"version": "1.9"}
        current_configs = {"PORT": "8081"}
        result = suggestion_rule_checker(current_configs, self.complex_rule, env_info, self.domain)
        self.assertEqual(result[0], CheckResult.OK)


class TestVersionComparison(unittest.TestCase):
    def test_version_comparison(self):
        version1 = "2.1"
        version2 = "2.2"
        self.assertLess(version1, version2)
        self.assertGreater(version2, version1)
