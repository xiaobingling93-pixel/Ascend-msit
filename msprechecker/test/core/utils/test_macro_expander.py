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

import re
import unittest
from unittest.mock import patch

from msprechecker.core.utils.macro_expander import MacroExpander


class TestMacroExpander(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            "server.host": "localhost",
            "server.port": 8080,
            "app.version": "1.2.3",
            "app.permissions": "644",
            "nested.config.value": 42,
            "empty": "",
        }
        self.context_path = "app.module"

    def test_expand_string_with_variable(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        result = expander.expand("Server: ${server.host}")
        self.assertEqual(result, "Server: localhost")

    def test_expand_string_with_version(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        result = expander.expand("Version: Version{app.version}")
        self.assertEqual(result, "Version: Version(app.version)")

    def test_expand_string_with_fileperm(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        result = expander.expand("Perm: FilePerm{app.permissions}")
        self.assertEqual(result, "Perm: FilePerm(app.permissions)")

    def test_expand_dict(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        input_dict = {
            "host": "${server.host}",
            "port": "${server.port}",
            "version": "Version{app.version}",
        }
        expected = {
            "host": "localhost",
            "port": "int(8080)",
            "version": "Version(app.version)",
        }
        result = expander.expand(input_dict)
        self.assertEqual(result, expected)

    def test_expand_list(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        input_list = ["${server.host}", "Version{app.version}", 123]
        expected = ["localhost", "Version(app.version)", 123]
        result = expander.expand(input_list)
        self.assertEqual(result, expected)

    def test_expand_non_string_value(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        self.assertEqual(expander.expand(42), 42)
        self.assertTrue(expander.expand(True))
        self.assertIsNone(expander.expand(None))

    def test_expand_empty_string(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        self.assertEqual(expander.expand(""), "")

    def test_expand_empty_variable_return_itself(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        self.assertEqual("${}", expander.expand("${}"))

    def test_expand_only_dots_variable_raises_error(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        with self.assertRaises(ValueError):
            expander.expand("${...}")

    def test_expand_nonexistent_variable_raises_error(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        with self.assertRaises(ValueError):
            expander.expand("${nonexistent.path}")

    def test_expand_relative_path(self):
        expander = MacroExpander(self.sample_config, "app.module.sub")
        result = expander.expand("${..version}")
        self.assertEqual(result, "1.2.3")

    def test_expand_too_many_relative_dots_raises_error(self):
        expander = MacroExpander(self.sample_config, "app.module")
        with self.assertRaises(ValueError):
            expander.expand("${...version}")

    def test_expand_complex_expression(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        result = expander.expand("${server.host}:${server.port} Version{app.version}")
        expected = "localhost:int(8080) Version(app.version)"
        self.assertEqual(result, expected)

    def test_expand_with_empty_config_value(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        result = expander.expand("${empty}")
        self.assertEqual(result, "")

    def test_expand_non_string_config_value(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        result = expander.expand("${nested.config.value}")
        self.assertEqual(result, "int(42)")

    def test_build_full_path_absolute(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        result = expander._build_full_path("server.host")
        self.assertEqual(result, "server.host")

    def test_build_full_path_relative_one_dot(self):
        expander = MacroExpander(self.sample_config, "app.module.sub")
        result = expander._build_full_path(".version")
        self.assertEqual(result, "app.module.version")

    def test_build_full_path_relative_two_dots(self):
        expander = MacroExpander(self.sample_config, "app.module.sub")
        result = expander._build_full_path("..version")
        self.assertEqual(result, "app.version")

    def test_build_full_path_relative_with_remaining(self):
        expander = MacroExpander(self.sample_config, "app.module.sub")
        result = expander._build_full_path("..config.value")
        self.assertEqual(result, "app.config.value")

    def test_build_full_path_invalid_relative_raises_error(self):
        expander = MacroExpander(self.sample_config, "app.module")
        with self.assertRaises(ValueError):
            expander._build_full_path("...version")

    def test_expand_var_non_string_value(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        with patch.object(expander, '_build_full_path', return_value="nested.config.value"):
            result = expander._expand_var(re.match(r"\$\{(.+?)\}", "${nested.config.value}"))
            self.assertEqual(result, "int(42)")

    def test_expand_var_string_value(self):
        expander = MacroExpander(self.sample_config, self.context_path)
        with patch.object(expander, '_build_full_path', return_value="server.host"):
            result = expander._expand_var(re.match(r"\$\{(.+?)\}", "${server.host}"))
            self.assertEqual(result, "localhost")
