"""
用例编号	方法名称 (snake_case)	测试场景描述	输入条件	预期结果	约束规则	临时资源处理	异常类型
1	test_validate_args_given_valid_readable_file_when_input_file_read_constraint_then_no_exception	可读文件符合input_file_read约束	创建只读临时文件(0o400)	不抛出异常	Rule.input_file_read	使用NamedTemporaryFile自动清理	-
2	test_validate_args_given_invalid_readable_file_when_input_file_read_constraint_then_raise_error	文件不存在违反input_file_read约束	"/nonexistent/file"	抛出ArgumentTypeError	Rule.input_file_read	无临时资源	argparse.ArgumentTypeError
3	test_validate_args_given_valid_executable_file_when_input_file_exec_constraint_then_no_exception	可执行文件符合input_file_exec约束	创建可执行临时文件(0o500)	不抛出异常	Rule.input_file_exec	使用NamedTemporaryFile自动清理	-
4	test_validate_args_given_invalid_executable_file_when_input_file_exec_constraint_then_raise_error	只读文件违反input_file_exec约束	创建只读临时文件(0o400)	抛出ArgumentTypeError	Rule.input_file_exec	使用NamedTemporaryFile自动清理	argparse.ArgumentTypeError
5	test_validate_args_given_valid_directory_when_input_dir_traverse_constraint_then_no_exception	有效目录符合input_dir_traverse约束	创建临时目录	不抛出异常	Rule.input_dir_traverse	使用TemporaryDirectory自动清理	-
6	test_validate_args_given_invalid_directory_when_input_dir_traverse_constraint_then_raise_error	不存在目录违反input_dir_traverse约束	"/nonexistent/directory"	抛出ArgumentTypeError	Rule.input_dir_traverse	无临时资源	argparse.ArgumentTypeError
7	test_validate_args_given_new_path_when_output_path_create_constraint_then_no_exception	新路径符合output_path_create约束	临时目录下不存在的文件路径	不抛出异常	Rule.output_path_create	使用TemporaryDirectory自动清理	-
8	test_validate_args_given_existing_path_when_output_path_create_constraint_then_raise_error	已存在路径违反output_path_create约束	已存在的临时文件	抛出ArgumentTypeError	Rule.output_path_create	使用NamedTemporaryFile自动清理	argparse.ArgumentTypeError
9	test_validate_args_given_root_user_when_output_path_overwrite_constraint_then_no_exception	root用户符合output_path_overwrite约束	模拟root用户操作已有文件	不抛出异常	Rule.output_path_overwrite	使用NamedTemporaryFile自动清理	-
10	test_validate_args_given_non_root_user_when_output_path_overwrite_constraint_then_raise_error	非root用户违反output_path_overwrite约束	非root用户操作已有文件	抛出ArgumentTypeError	Rule.output_path_overwrite	使用NamedTemporaryFile自动清理	argparse.ArgumentTypeError
11	test_validate_args_given_valid_path_when_output_path_write_constraint_then_no_exception	可写路径符合output_path_write约束	临时目录下新文件路径	不抛出异常	Rule.output_path_write	使用TemporaryDirectory自动清理	-
12	test_validate_args_given_invalid_path_when_output_path_write_constraint_then_raise_error	无效路径违反output_path_write约束	"/nonexistent/path/to/file"	抛出ArgumentTypeError	Rule.output_path_write	无临时资源	argparse.ArgumentTypeError
13	test_validate_args_given_root_user_when_output_dir_constraint_then_no_exception	root用户符合output_dir约束	模拟root用户操作目录	不抛出异常	Rule.output_dir	使用TemporaryDirectory自动清理	-
14	test_validate_args_given_non_root_user_when_output_dir_constraint_then_raise_error	非root用户违反output_dir约束	非root用户操作目录	抛出ArgumentTypeError	Rule.output_dir	使用TemporaryDirectory自动清理	argparse.ArgumentTypeError
15	test_validate_args_given_invalid_path_when_silent_false_then_raise_invalid_parameter_error	silent=False时无效路径抛出特定异常	"/nonexistent/file"	抛出InvalidParameterError	Rule.input_file_read+silent=False	无临时资源	InvalidParameterError
16	test_validate_args_given_non_string_input_when_any_constraint_then_raise_argument_type_error	非字符串输入违反类型约束	整数123	抛出ArgumentTypeError	任意规则	无临时资源	argparse.ArgumentTypeError
"""

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

import os
import argparse
import unittest
import tempfile
from unittest.mock import patch

from msguard import validate_args, Rule, InvalidParameterError


class TestValidateArgs(unittest.TestCase):
    """Unit tests for validate_args decorator"""

    def test_validate_args_given_valid_readable_file_when_input_file_read_constraint_then_no_exception(self):
        """当提供可读文件且符合input_file_read约束时，不应抛出异常"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            os.chmod(tmp_file.name, 0o400)  # 设置为只读
            validator = validate_args(Rule.input_file_read)
            validator(tmp_file.name)  # 预期不会抛出异常

    def test_validate_args_given_invalid_readable_file_when_input_file_read_constraint_then_raise_error(self):
        """当文件不存在且不符合input_file_read约束时，应抛出ArgumentTypeError"""
        validator = validate_args(Rule.input_file_read)
        with self.assertRaises(argparse.ArgumentTypeError):
            validator("/nonexistent/file")  # 文件不存在，预期抛出异常

    def test_validate_args_given_valid_executable_file_when_input_file_exec_constraint_then_no_exception(self):
        """当提供可执行文件且符合input_file_exec约束时，不应抛出异常"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            os.chmod(tmp_file.name, 0o500)  # 设置为可读可执行
            validator = validate_args(Rule.input_file_exec)
            validator(tmp_file.name)  # 预期不会抛出异常

    @unittest.skipIf(os.geteuid() == 0, "root用户可执行任何文件")
    def test_validate_args_given_invalid_executable_file_when_input_file_exec_constraint_then_raise_error(self):
        """当文件不可执行且不符合input_file_exec约束时，应抛出ArgumentTypeError"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            os.chmod(tmp_file.name, 0o400)  # 设置为只读
            validator = validate_args(Rule.input_file_exec)
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(tmp_file.name)  # 文件不可执行，预期抛出异常

    def test_validate_args_given_valid_directory_when_input_dir_traverse_constraint_then_no_exception(self):
        """当提供有效目录且符合input_dir_traverse约束时，不应抛出异常"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validator = validate_args(Rule.input_dir_traverse)
            validator(tmp_dir)  # 预期不会抛出异常

    def test_validate_args_given_invalid_directory_when_input_dir_traverse_constraint_then_raise_error(self):
        """当目录不存在且不符合input_dir_traverse约束时，应抛出ArgumentTypeError"""
        validator = validate_args(Rule.input_dir_traverse)
        with self.assertRaises(argparse.ArgumentTypeError):
            validator("/nonexistent/directory")  # 目录不存在，预期抛出异常

    def test_validate_args_given_new_path_when_output_path_create_constraint_then_no_exception(self):
        """当路径不存在且符合output_path_create约束时，不应抛出异常"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_file = os.path.join(tmp_dir, "newfile.txt")
            validator = validate_args(Rule.output_path_create)
            validator(new_file)  # 预期不会抛出异常

    def test_validate_args_given_existing_path_when_output_path_create_constraint_then_raise_error(self):
        """当路径已存在且不符合output_path_create约束时，应抛出ArgumentTypeError"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            validator = validate_args(Rule.output_path_create)
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(tmp_file.name)  # 文件已存在，预期抛出异常

    @patch('os.getuid', return_value=0)
    def test_validate_args_given_root_user_when_output_path_overwrite_constraint_then_no_exception(self, mock_getuid):
        """当root用户尝试覆盖文件且符合output_path_overwrite约束时，不应抛出异常"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            validator = validate_args(Rule.output_path_overwrite)
            validator(tmp_file.name)  # root用户可覆盖，预期不会抛出异常

    @unittest.skipIf(os.geteuid() == 0, "root用户可以覆盖任何文件")
    def test_validate_args_given_non_root_user_when_output_path_overwrite_constraint_then_raise_error(self):
        """当非root用户尝试覆盖文件且不符合output_path_overwrite约束时，应抛出ArgumentTypeError"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            validator = validate_args(Rule.output_path_overwrite)
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(tmp_file.name)  # 非root用户不可覆盖，预期抛出异常

    def test_validate_args_given_valid_path_when_output_path_write_constraint_then_no_exception(self):
        """当路径可写且符合output_path_write约束时，不应抛出异常"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_file = os.path.join(tmp_dir, "newfile.txt")
            validator = validate_args(Rule.output_path_write)
            validator(new_file)  # 预期不会抛出异常

    def test_validate_args_given_invalid_path_when_output_path_write_constraint_then_raise_error(self):
        """当路径不可写且不符合output_path_write约束时，应抛出ArgumentTypeError"""
        validator = validate_args(Rule.output_path_write)
        with self.assertRaises(argparse.ArgumentTypeError):
            validator("/nonexistent/path/to/file")  # 父目录不存在，预期抛出异常

    @patch('os.getuid', return_value=0)
    def test_validate_args_given_root_user_when_output_dir_constraint_then_no_exception(self, mock_getuid):
        """当root用户尝试创建目录且符合output_dir约束时，不应抛出异常"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validator = validate_args(Rule.output_dir)
            validator(tmp_dir)  # root用户可创建目录，预期不会抛出异常

    @unittest.skipIf(os.geteuid() == 0, "root用户可以创建任何目录")
    def test_validate_args_given_non_root_user_when_output_dir_constraint_then_raise_error(self):
        """当非root用户尝试创建目录且不符合output_dir约束时，应抛出ArgumentTypeError"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validator = validate_args(Rule.output_dir)
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(tmp_dir)  # 非root用户不可创建目录，预期抛出异常

    def test_validate_args_given_invalid_path_when_silent_false_then_raise_invalid_parameter_error(self):
        """当silent=False且路径无效时，应抛出InvalidParameterError"""
        validator = validate_args(Rule.input_file_read, silent=False)
        with self.assertRaises(InvalidParameterError):
            validator("/nonexistent/file")  # 文件不存在且silent=False，预期抛出特定异常

    def test_validate_args_given_non_string_input_when_any_constraint_then_raise_argument_type_error(self):
        """当输入不是字符串时，无论何种约束都应抛出ArgumentTypeError"""
        validator = validate_args(Rule.input_file_read)
        with self.assertRaises(argparse.ArgumentTypeError):
            validator(123)  # 非字符串输入，预期抛出异常
