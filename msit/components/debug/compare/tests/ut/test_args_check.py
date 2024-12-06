# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
from unittest.mock import patch, MagicMock
import os
import argparse
import re

from components.utils.file_open_check import FileStat
from components.debug.compare.msquickcmp.common.args_check import (
    check_model_path_legality,
    check_om_path_legality,
    check_weight_path_legality,
    check_input_path_legality,
    check_output_path_legality,
    check_dict_kind_string,
    check_device_range_valid,
    check_number_list,
    check_dym_range_string,
    str2bool,
    safe_string,
    is_saved_model_valid,
    valid_json_file_or_dir,
    check_cann_path_legality,
    check_fusion_cfg_path_legality,
    check_quant_json_path_legality
)
STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")

class TestIsSavedModelValid(unittest.TestCase):

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    def test_valid_saved_model(self, mock_isfile, mock_isdir):
        # 模拟合法的 saved_model 文件夹
        mock_isdir.side_effect = lambda path: path == "valid_directory" or path == "valid_directory/variables"
        mock_isfile.side_effect = lambda path: path == "valid_directory/saved_model.pb"

        result = is_saved_model_valid("valid_directory")
        self.assertTrue(result)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    def test_missing_saved_model_pb(self, mock_isfile, mock_isdir):
        # 模拟 missing saved_model.pb 文件
        mock_isdir.side_effect = lambda path: path == "valid_directory"
        mock_isfile.side_effect = lambda path: False  # saved_model.pb 不存在

        result = is_saved_model_valid("valid_directory")
        self.assertFalse(result)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    def test_missing_variables_directory(self, mock_isfile, mock_isdir):
        # 模拟 missing variables 目录
        mock_isdir.side_effect = lambda path: path == "valid_directory"
        mock_isfile.side_effect = lambda path: path == "valid_directory/saved_model.pb"

        result = is_saved_model_valid("valid_directory")
        self.assertFalse(result)


class TestCheckModelPathLegality(unittest.TestCase):
    @patch('os.path.isdir')
    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_file_type(self, MockFileStat, MockIsDir):
        # 模拟文件类型不合法
        MockIsDir.return_value = False
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = True
        mock_file.is_legal_file_type.return_value = False
        mock_file.is_legal_file_size.return_value = True

        with self.assertRaises(argparse.ArgumentTypeError):
            check_model_path_legality("invalid_model.txt")

    @patch('os.path.isdir')
    @patch('components.utils.file_open_check.FileStat')
    def test_file_size_exceed(self, MockFileStat, MockIsDir):
        # 模拟文件大小超出限制
        MockIsDir.return_value = False
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = True
        mock_file.is_legal_file_type.return_value = True
        mock_file.is_legal_file_size.return_value = False

        with self.assertRaises(argparse.ArgumentTypeError):
            check_model_path_legality("large_model.onnx")

    @patch('os.path.isdir')
    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_directory(self, MockFileStat, MockIsDir):
        # 模拟无效的文件夹路径
        MockIsDir.return_value = True
        MockFileStat.return_value.is_basically_legal.return_value = False

        with self.assertRaises(argparse.ArgumentTypeError):
            check_model_path_legality("invalid_directory")

    @patch('os.path.isdir')
    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_path(self, MockFileStat, MockIsDir):
        # 模拟无效路径（不存在的路径）
        MockIsDir.return_value = False
        MockFileStat.side_effect = Exception("Path not found")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_model_path_legality("non_existent_path")


class TestCheckOmPathLegality(unittest.TestCase):
    @patch('os.path.isdir')
    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_file_type(self, MockFileStat, MockIsDir):
        # 模拟文件类型不合法（如 .txt 文件）
        MockIsDir.return_value = False
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = True
        mock_file.is_legal_file_type.return_value = False
        mock_file.is_legal_file_size.return_value = True

        with self.assertRaises(argparse.ArgumentTypeError):
            check_om_path_legality("invalid_model.txt")

    @patch('os.path.isdir')
    @patch('components.utils.file_open_check.FileStat')
    def test_file_size_exceed(self, MockFileStat, MockIsDir):
        # 模拟文件大小超出限制
        MockIsDir.return_value = False
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = True
        mock_file.is_legal_file_type.return_value = True
        mock_file.is_legal_file_size.return_value = False

        with self.assertRaises(argparse.ArgumentTypeError):
            check_om_path_legality("large_model.om")

    @patch('os.path.isdir')
    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_directory(self, MockFileStat, MockIsDir):
        # 模拟无效的文件夹路径
        MockIsDir.return_value = True
        MockFileStat.return_value.is_basically_legal.return_value = False

        with self.assertRaises(argparse.ArgumentTypeError):
            check_om_path_legality("invalid_directory")

    @patch('os.path.isdir')
    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_path(self, MockFileStat, MockIsDir):
        # 模拟无效路径（不存在的路径）
        MockIsDir.return_value = False
        MockFileStat.side_effect = Exception("Path not found")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_om_path_legality("non_existent_path")

class TestCheckWeightPathLegality(unittest.TestCase):
    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_file_type(self, MockFileStat):
        # 模拟无效文件类型（如 .txt 文件）
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = True
        mock_file.is_legal_file_type.return_value = False
        mock_file.is_legal_file_size.return_value = True

        with self.assertRaises(argparse.ArgumentTypeError):
            check_weight_path_legality("invalid_model.txt")

    @patch('components.utils.file_open_check.FileStat')
    def test_file_size_exceed(self, MockFileStat):
        # 模拟文件大小超出限制
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = True
        mock_file.is_legal_file_type.return_value = True
        mock_file.is_legal_file_size.return_value = False

        with self.assertRaises(argparse.ArgumentTypeError):
            check_weight_path_legality("large_model.caffemodel")

    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_path(self, MockFileStat):
        # 模拟无效路径或文件（不存在的文件路径）
        MockFileStat.side_effect = Exception("Path not found")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_weight_path_legality("non_existent_path.caffemodel")

    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_file_permissions(self, MockFileStat):
        # 模拟无法读取的文件（权限问题）
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = False  # 权限问题，无法读取文件

        with self.assertRaises(argparse.ArgumentTypeError):
            check_weight_path_legality("restricted_model.caffemodel")

class TestCheckInputPathLegality(unittest.TestCase):
    @patch('components.utils.file_open_check.FileStat')
    def test_empty_input(self, MockFileStat):
        # 测试空输入路径
        result = check_input_path_legality("")
        self.assertEqual(result, "")

    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_input_path(self, MockFileStat):
        # 模拟无效的输入路径
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = False  # 无法读取

        with self.assertRaises(argparse.ArgumentTypeError):
            check_input_path_legality("invalid_path")

    @patch('components.utils.file_open_check.FileStat')
    def test_multiple_input_paths_some_invalid(self, MockFileStat):
        # 模拟多个路径，部分路径无效
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = True

        # 设置第一个路径合法，第二个路径无效
        mock_file.is_basically_legal.side_effect = [True, False]

        with self.assertRaises(argparse.ArgumentTypeError):
            check_input_path_legality("valid_path,invalid_path")

    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_path_exception(self, MockFileStat):
        # 模拟路径无效（如不存在的文件路径）
        MockFileStat.side_effect = Exception("Path not found")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_input_path_legality("non_existent_path")

class TestCheckCannPathLegality(unittest.TestCase):

    @patch('components.utils.file_open_check.is_legal_args_path_string')
    def test_valid_cann_path(self, mock_is_legal_args_path_string):
        # 模拟合法路径
        mock_is_legal_args_path_string.return_value = True

        # 测试合法路径
        result = check_cann_path_legality("valid/cann/path")
        self.assertEqual(result, "valid/cann/path")

    @patch('components.utils.file_open_check.is_legal_args_path_string')
    def test_invalid_cann_path(self, mock_is_legal_args_path_string):
        # 模拟非法路径
        mock_is_legal_args_path_string.return_value = False

        # 测试非法路径，期望抛出 ArgumentTypeError
        with self.assertRaises(argparse.ArgumentTypeError):
            check_cann_path_legality("invalid/cann/path&")

class TestCheckOutputPathLegality(unittest.TestCase):

    @patch('components.utils.file_open_check.FileStat')
    def test_valid_output_path(self, MockFileStat):
        # 模拟合法路径：文件路径可写
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.return_value = True  # 可写权限

        # 测试合法路径
        result = check_output_path_legality("valid/output/path")
        self.assertEqual(result, "valid/output/path")

    @patch('components.utils.file_open_check.FileStat')
    def test_empty_output_path(self, MockFileStat):
        # 测试空路径
        result = check_output_path_legality("")
        self.assertEqual(result, "")

    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_output_path(self, MockFileStat):
        # 模拟路径无效（如不存在的文件路径）
        MockFileStat.side_effect = Exception("Path not found")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_output_path_legality("non_existent_output_path&")

class TestValidJsonFileOrDir(unittest.TestCase):

    @patch('components.utils.file_open_check.FileStat')
    def test_empty_input_path(self, MockFileStat):
        # 测试空路径
        result = valid_json_file_or_dir("")
        self.assertEqual(result, "")

    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_file_type(self, MockFileStat):
        # 模拟文件类型不合法
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_dir = False
        mock_file.is_basically_legal.return_value = True
        mock_file.is_legal_file_type.return_value = False  # 非 JSON 文件

        with self.assertRaises(argparse.ArgumentTypeError):
            valid_json_file_or_dir("invalid/file.txt")

    @patch('components.utils.file_open_check.FileStat')
    def test_invalid_file_size(self, MockFileStat):
        # 模拟文件大小不合法
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_dir = False
        mock_file.is_basically_legal.return_value = True
        mock_file.is_legal_file_type.return_value = True  # JSON 文件
        mock_file.is_legal_file_size.return_value = False  # 文件大小超出限制

        with self.assertRaises(argparse.ArgumentTypeError):
            valid_json_file_or_dir("invalid/large_file.json")

    @patch('components.utils.file_open_check.FileStat')
    def test_unreadable_path(self, MockFileStat):
        # 模拟路径不可读取（权限问题）
        mock_file = MagicMock()
        MockFileStat.return_value = mock_file
        mock_file.is_basically_legal.side_effect = Exception("No read permission")

        with self.assertRaises(argparse.ArgumentTypeError):
            valid_json_file_or_dir("unreadable/file.json")

class TestCheckDictKindString(unittest.TestCase):

    def test_empty_string(self):
        # 测试空字符串
        result = check_dict_kind_string("")
        self.assertEqual(result, "")

    def test_valid_string(self):
        # 测试合法字符串
        valid_input = "input_name1:1,224,224,3;input_name2:3,300"
        result = check_dict_kind_string(valid_input)
        self.assertEqual(result, valid_input)

    def test_invalid_string_with_illegal_character(self):
        # 测试包含非法字符的字符串 (如空格)
        invalid_input = "input_name1:1,224,224,3 ;input_name2:3,300"

        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_dict_kind_string(invalid_input)

        # 验证错误信息
        self.assertEqual(str(cm.exception), f'dym string "{invalid_input}" is not a legal string')

    def test_invalid_string_with_special_character(self):
        # 测试包含非法字符的字符串 (如特殊符号)
        invalid_input = "input_name1:1,224,224,3@input_name2:3,300"

        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_dict_kind_string(invalid_input)

        # 验证错误信息
        self.assertEqual(str(cm.exception), f'dym string "{invalid_input}" is not a legal string')

    def test_valid_string_with_special_characters(self):
        # 测试包含合法字符的字符串（如 ":", ",", ";", "." 等）
        valid_input = "input_name1:1,224,224,3;input_name2:3,300.input_name3:5,300"
        result = check_dict_kind_string(valid_input)
        self.assertEqual(result, valid_input)

class TestCheckDeviceRangeValid(unittest.TestCase):

    def test_valid_value_within_range(self):
        # 测试合法值在范围内
        valid_values = [0, 100, 255]
        for value in valid_values:
            check_device_range_valid(value)

    def test_value_below_minimum(self):
        # 测试小于最小值的值
        invalid_value = -1
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_device_range_valid(invalid_value)
        self.assertEqual(str(cm.exception), "device:-1 is invalid. valid value range is [0, 255]")

    def test_value_above_maximum(self):
        # 测试大于最大值的值
        invalid_value = 256
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_device_range_valid(invalid_value)
        self.assertEqual(str(cm.exception), "device:256 is invalid. valid value range is [0, 255]")

    def test_non_integer_value(self):
        # 测试非整数值
        invalid_value = "abc"
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_device_range_valid(invalid_value)
        self.assertEqual(str(cm.exception), "input:abc is illegal.Please check")

class TestCheckNumberList(unittest.TestCase):

    def test_valid_number_list(self):
        # 测试有效的数字列表
        valid_values = [
            "1241414,124141,124424",
            "1,2,3,4,5",
            "100,200,300",
            "0"
        ]
        for value in valid_values:
            result = check_number_list(value)
            self.assertEqual(result, value)  # 确保返回值与输入一致

    def test_empty_input(self):
        # 测试空字符串输入
        empty_value = ""
        result = check_number_list(empty_value)
        self.assertEqual(result, empty_value)  # 空字符串应直接返回

    def test_none_input(self):
        # 测试None输入
        none_value = None
        result = check_number_list(none_value)
        self.assertEqual(result, none_value)  # None应直接返回

    def test_invalid_number_list_with_letters(self):
        # 测试包含字母的非法数字列表
        invalid_value = "124141a,124141,124424"
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_number_list(invalid_value)
        self.assertEqual(str(cm.exception), 'output size "124141a" is not a legal string')

    def test_invalid_number_list_with_special_characters(self):
        # 测试包含特殊字符的非法数字列表
        invalid_value = "124141,@124141,124424"
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_number_list(invalid_value)
        self.assertEqual(str(cm.exception), 'output size "@124141" is not a legal string')

    def test_invalid_number_list_with_spaces(self):
        # 测试包含空格的非法数字列表
        invalid_value = "124141 124141,124424"
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_number_list(invalid_value)
        self.assertEqual(str(cm.exception), 'output size "124141 124141" is not a legal string')

class TestCheckDymRangeString(unittest.TestCase):

    def test_valid_dym_range_string(self):
        # 测试有效的dym range字符串
        valid_values = [
            "input1:1,224,224,3;input2:3,300",  # 有效的dym range字符串
            "input1:1,224,224,3~input2:3,300",  # 包含波浪符号
            "input1:1,224,224,3;input2:3,300:255",  # 含有冒号
            "input1:0,0,0,0",  # 简单的有效字符串
            "input123-456_789:0~1"  # 混合字母、数字、符号的有效字符串
        ]
        for value in valid_values:
            result = check_dym_range_string(value)
            self.assertEqual(result, value)  # 确保返回值与输入一致

    def test_empty_input(self):
        # 测试空字符串输入
        empty_value = ""
        result = check_dym_range_string(empty_value)
        self.assertEqual(result, empty_value)  # 空字符串应直接返回

    def test_none_input(self):
        # 测试None输入
        none_value = None
        result = check_dym_range_string(none_value)
        self.assertEqual(result, none_value)  # None应直接返回

    def test_invalid_dym_range_string_with_spaces(self):
        # 测试包含空格的非法dym range字符串
        invalid_value = "input1:1,224,224,3 ;input2:3,300"
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_dym_range_string(invalid_value)
        self.assertEqual(str(cm.exception), 'dym range string "input1:1,224,224,3 ;input2:3,300" is not a legal string')

    def test_invalid_dym_range_string_with_special_characters(self):
        # 测试包含非法特殊字符的dym range字符串
        invalid_value = "input1:1,224,224,3@input2:3,300"
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_dym_range_string(invalid_value)
        self.assertEqual(str(cm.exception), 'dym range string "input1:1,224,224,3@input2:3,300" is not a legal string')

    def test_invalid_dym_range_string_with_non_ascii(self):
        # 测试包含非ASCII字符的dym range字符串
        invalid_value = "input1:1,224,224,3;输入2:3,300"
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            check_dym_range_string(invalid_value)
        self.assertEqual(str(cm.exception), 'dym range string "input1:1,224,224,3;输入2:3,300" is not a legal string')

class TestCheckFusionCfgPathLegality(unittest.TestCase):

    def setUp(self):
        # 模拟 FileStat 类
        self.mock_file_stat = MagicMock()

    def test_empty_input(self):
        # 测试空字符串输入
        empty_value = ""
        result = check_fusion_cfg_path_legality(empty_value)
        self.assertEqual(result, empty_value)

    def test_none_input(self):
        # 测试None输入
        none_value = None
        result = check_fusion_cfg_path_legality(none_value)
        self.assertEqual(result, none_value)

    def test_invalid_fusion_cfg_path_with_permission_error(self):
        # 测试路径无读取权限
        invalid_value = "/path/to/invalid/fusion/config.cfg"
        self.mock_file_stat.is_basically_legal.return_value = False

        with unittest.mock.patch('components.utils.file_open_check.FileStat', return_value=self.mock_file_stat):
            with self.assertRaises(argparse.ArgumentTypeError) as cm:
                check_fusion_cfg_path_legality(invalid_value)
            self.assertEqual(str(cm.exception), f"fusion switch file path:{invalid_value} is illegal. Please check.")

    def test_invalid_fusion_cfg_path_with_invalid_type(self):
        # 测试路径文件类型不合法
        invalid_value = "/path/to/fusion/config.txt"
        self.mock_file_stat.is_basically_legal.return_value = True
        self.mock_file_stat.is_legal_file_type.return_value = False

        with unittest.mock.patch('components.utils.file_open_check.FileStat', return_value=self.mock_file_stat):
            with self.assertRaises(argparse.ArgumentTypeError) as cm:
                check_fusion_cfg_path_legality(invalid_value)
            self.assertEqual(str(cm.exception), f"fusion switch file path:{invalid_value} is illegal. Please check.")

    def test_invalid_fusion_cfg_path_with_large_size(self):
        # 测试文件大小超过限制
        invalid_value = "/path/to/fusion/large_config.cfg"
        self.mock_file_stat.is_basically_legal.return_value = True
        self.mock_file_stat.is_legal_file_type.return_value = True
        self.mock_file_stat.is_legal_file_size.return_value = False

        with unittest.mock.patch('components.utils.file_open_check.FileStat', return_value=self.mock_file_stat):
            with self.assertRaises(argparse.ArgumentTypeError) as cm:
                check_fusion_cfg_path_legality(invalid_value)
            self.assertEqual(str(cm.exception), f"fusion switch file path:{invalid_value} is illegal. Please check.")

class TestCheckQuantJsonPathLegality(unittest.TestCase):

    def setUp(self):
        # 模拟 FileStat 类
        self.mock_file_stat = MagicMock()

    def test_empty_input(self):
        # 测试空字符串输入
        empty_value = ""
        result = check_quant_json_path_legality(empty_value)
        self.assertEqual(result, empty_value)

    def test_none_input(self):
        # 测试None输入
        none_value = None
        result = check_quant_json_path_legality(none_value)
        self.assertEqual(result, none_value)

    def test_invalid_quant_json_path_with_permission_error(self):
        # 测试路径无读取权限
        invalid_value = "/path/to/invalid/quantization/config.json"
        self.mock_file_stat.is_basically_legal.return_value = False

        with unittest.mock.patch('components.utils.file_open_check.FileStat', return_value=self.mock_file_stat):
            with self.assertRaises(argparse.ArgumentTypeError) as cm:
                check_quant_json_path_legality(invalid_value)
            self.assertEqual(str(cm.exception), f"quant file path:{invalid_value} is illegal. Please check.")

    def test_invalid_quant_json_path_with_invalid_type(self):
        # 测试路径文件类型不合法
        invalid_value = "/path/to/quantization/config.txt"
        self.mock_file_stat.is_basically_legal.return_value = True
        self.mock_file_stat.is_legal_file_type.return_value = False

        with unittest.mock.patch('components.utils.file_open_check.FileStat', return_value=self.mock_file_stat):
            with self.assertRaises(argparse.ArgumentTypeError) as cm:
                check_quant_json_path_legality(invalid_value)
            self.assertEqual(str(cm.exception), f"quant file path:{invalid_value} is illegal. Please check.")

    def test_invalid_quant_json_path_with_large_size(self):
        # 测试文件大小超过限制
        invalid_value = "/path/to/quantization/large_config.json"
        self.mock_file_stat.is_basically_legal.return_value = True
        self.mock_file_stat.is_legal_file_type.return_value = True
        self.mock_file_stat.is_legal_file_size.return_value = False

        with unittest.mock.patch('components.utils.file_open_check.FileStat', return_value=self.mock_file_stat):
            with self.assertRaises(argparse.ArgumentTypeError) as cm:
                check_quant_json_path_legality(invalid_value)
            self.assertEqual(str(cm.exception), f"quant file path:{invalid_value} is illegal. Please check.")

class TestSafeString(unittest.TestCase):

    def test_empty_input(self):
        # 测试空字符串输入
        empty_value = ""
        result = safe_string(empty_value)
        self.assertEqual(result, empty_value)

    def test_none_input(self):
        # 测试None输入
        none_value = None
        result = safe_string(none_value)
        self.assertEqual(result, none_value)

    def test_valid_string(self):
        # 测试有效字符串
        valid_value = "valid_string123"
        result = safe_string(valid_value)
        self.assertEqual(result, valid_value)

    def test_string_with_invalid_character(self):
        # 测试包含无效字符的字符串
        invalid_value = "invalid$string"
        with self.assertRaises(ValueError) as cm:
            safe_string(invalid_value)
        self.assertEqual(str(cm.exception), "String parameter contains invalid characters.")

    def test_string_with_legal_special_characters(self):
        # 测试包含合法特殊字符的字符串
        valid_value = "String parameter contains invalid characters."
        result = safe_string(valid_value)
        self.assertEqual(result, valid_value)


class TestStr2Bool(unittest.TestCase):

    def test_boolean_true(self):
        # 测试输入布尔值 True
        self.assertTrue(str2bool(True))

    def test_boolean_false(self):
        # 测试输入布尔值 False
        self.assertFalse(str2bool(False))

    def test_string_true_values(self):
        # 测试表示True的字符串
        true_values = ['yes', 'true', 't', 'y', '1']
        for value in true_values:
            self.assertTrue(str2bool(value))

    def test_string_false_values(self):
        # 测试表示False的字符串
        false_values = ['no', 'false', 'f', 'n', '0']
        for value in false_values:
            self.assertFalse(str2bool(value))

    def test_invalid_value(self):
        # 测试无效的字符串输入，应该抛出ArgumentTypeError
        invalid_values = ['maybe', 'invalid', '123', 'anything']
        for value in invalid_values:
            with self.assertRaises(argparse.ArgumentTypeError) as cm:
                str2bool(value)
            self.assertEqual(str(cm.exception), 'Boolean value expected true, 1, false, 0 with case insensitive.')
