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

from msprechecker.core.utils.version import Version
from msprechecker.core.utils.perm import FilePerm
from msprechecker.core.utils.compiler import Compiler


class TestCompiler(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler

    def test_compile_simple_arithmetic(self):
        self.assertEqual(self.compiler.compile("1 + 2"), 3)
        self.assertEqual(self.compiler.compile("5 * 3"), 15)
        self.assertEqual(self.compiler.compile("10 - 4"), 6)
        self.assertEqual(self.compiler.compile("10 / 2"), 5.0)
        self.assertEqual(self.compiler.compile("10 // 3"), 3)
        self.assertEqual(self.compiler.compile("10 % 3"), 1)
        self.assertEqual(self.compiler.compile("2 ** 3"), 8)

    def test_compile_operator_precedence(self):
        self.assertEqual(self.compiler.compile("1 + 2 * 3"), 7)
        self.assertEqual(self.compiler.compile("(1 + 2) * 3"), 9)
        self.assertEqual(self.compiler.compile("2 * 3 + 4"), 10)
        self.assertEqual(self.compiler.compile("2 * (3 + 4)"), 14)
        self.assertEqual(self.compiler.compile("2 ** 3 * 4"), 32)
        self.assertEqual(self.compiler.compile("2 * 3 ** 4"), 162)

    def test_compile_function_calls(self):
        self.assertEqual(self.compiler.compile("int(5)"), 5)
        self.assertEqual(self.compiler.compile("float(5)"), 5.0)
        self.assertIsInstance(self.compiler.compile("Version(1.2.3)"), Version)
        self.assertIsInstance(self.compiler.compile("FilePerm(644)"), FilePerm)

    def test_compile_version_conflict(self):
        with self.assertRaises(SyntaxError):
            self.compiler.compile("Version(1.2.3) + 1")
        with self.assertRaises(SyntaxError):
            self.compiler.compile("Version(1.2.3) * 2")
        with self.assertRaises(SyntaxError):
            self.compiler.compile("1 + Version(1.2.3)")

    def test_compile_invalid_expressions(self):
        with self.assertRaises(ValueError):
            self.compiler.compile("1 + (2 * 3")
        with self.assertRaises(ValueError):
            self.compiler.compile("1 + 2 * 3)")
        self.assertEqual(self.compiler.compile("1 & 2"), "1 & 2")

    def test_compile_non_string_input(self):
        self.assertEqual(self.compiler.compile(42), 42)
        self.assertEqual(self.compiler.compile(3.14), 3.14)
        self.assertTrue(self.compiler.compile(True))
        self.assertEqual(self.compiler.compile(None), None)

    def test_compile_list_input(self):
        input_list = ["1 + 2", "3 * 4", "Version(1.2.3)"]
        expected = [3, 12, Version('1.2.3')]
        self.assertEqual(self.compiler.compile(input_list), expected)

    def test_compile_dict_input(self):
        input_dict = {"a": "1 + 2", "b": "3 * 4", "c": "Version(1.2.3)"}
        expected = {"a": 3, "b": 12, "c": Version('1.2.3')}
        self.assertEqual(self.compiler.compile(input_dict), expected)

    def test_compile_whitespace_handling(self):
        self.assertEqual(self.compiler.compile("  1  +  2  "), 3)
        self.assertEqual(self.compiler.compile("\t1\t+\t2\t"), 3)

    def test_compile_returns_original_on_error(self):
        self.assertEqual(self.compiler.compile("invalid expression"), "invalid expression")
        self.assertEqual(self.compiler.compile("1 + "), "1 + ")

    def test_split_tokens(self):
        tokens = self.compiler._split_tokens("1 + 2 * 3")
        expected = [
            ('NUMBER', '1'),
            ('OP', '+'),
            ('NUMBER', '2'),
            ('OP', '*'),
            ('NUMBER', '3')
        ]
        self.assertEqual(tokens, expected)

    def test_convert_tokens_to_rpn(self):
        tokens = [
            ('NUMBER', '1'),
            ('OP', '+'),
            ('NUMBER', '2'),
            ('OP', '*'),
            ('NUMBER', '3')
        ]
        rpn = self.compiler._convert_tokens_to_rpn(tokens)
        expected = [1, 2, 3, '*', '+']
        self.assertEqual(rpn, expected)

    def test_evaluate_rpn(self):
        rpn = [1, 2, 3, '*', '+']
        result = self.compiler._evaluate_rpn(rpn)
        self.assertEqual(result, 7)

    def test_evaluate_rpn_single_token(self):
        rpn = [42]
        result = self.compiler._evaluate_rpn(rpn)
        self.assertEqual(result, 42)

    def test_evaluate_rpn_version(self):
        version = Version('1.2.3')
        rpn = [version]
        result = self.compiler._evaluate_rpn(rpn)
        self.assertEqual(result, version)
