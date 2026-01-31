# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
from argparse import ArgumentTypeError
import unittest
from unittest.mock import patch, MagicMock
import pytest

from auto_optimizer.common.args_check import (
    check_in_path_legality,
    check_in_model_path_legality,
    check_soc,
    check_range,
    check_min_num_1,
    check_min_num_2,
    check_shapes_string,
    check_dtypes_string,
    check_io_string,
    check_nodes_string,
    check_single_node_string,
    check_normal_string,
    check_shapes_range_string,
    check_ints_string,
    check_path_string,
)


class TestPathLegalityFunctions(unittest.TestCase):

    @patch('components.utils.file_open_check.FileStat')
    def test_check_in_path_legality_no_read_permission(self, patched_FileStat):
        path_value = "/invalid/path"
        mock_file_stat = MagicMock()
        mock_file_stat.is_basically_legal.return_value = False
        patched_FileStat.return_value = mock_file_stat

        with pytest.raises(ArgumentTypeError) as excinfo:
            check_in_path_legality(path_value)

        assert str(excinfo.value) == "The current input file does not have right read permission, please check."

    @patch('components.utils.file_open_check.FileStat')
    def test_check_in_model_path_legality_no_read_permission(self, patched_FileStat):
        path_value = "/path/to/model.onnx"
        mock_file_stat = MagicMock()
        mock_file_stat.is_basically_legal.return_value = False
        patched_FileStat.return_value = mock_file_stat

        with pytest.raises(ArgumentTypeError) as excinfo:
            check_in_model_path_legality(path_value)

        assert str(excinfo.value) == "The current input model file does not have right read permission, please check."

    def test_check_soc_non_digit_string(self):
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_soc("abc123")
        assert str(excinfo.value) == "The input 'device' param is not valid."
    
    def test_check_range_non_digit_string(self):
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_range("abc123")
        assert str(excinfo.value) == "The input 'processes' param is not valid."

    def test_check_range_digit_string_valid(self):
        value = "10"
        result = check_range(value)
        assert result == 10

    def test_check_range_digit_string_below_range(self):
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_range("0")
        assert str(excinfo.value) == "0 is not a valid value. Range 1 ~ 64."

    def test_check_range_digit_string_above_range(self):
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_range("65")
        assert str(excinfo.value) == "65 is not a valid value. Range 1 ~ 64."

    def test_check_range_empty_string(self):
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_range("")
        assert str(excinfo.value) == "The input 'processes' param is not valid."

    def test_check_min_num_1_non_digit_string(self):
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_min_num_1("abc123")
        assert str(excinfo.value) == "The input 'loop' param is not valid."

    def test_check_min_num_1_digit_string_valid(self):
        value = "10"
        result = check_min_num_1(value)
        assert result == 10

    def test_check_min_num_1_digit_string_less_than_1(self):
        value = "0"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_min_num_1(value)
        assert str(excinfo.value) == "0 is not a valid value. Minimum value 1."

    def test_check_min_num_1_empty_string(self):
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_min_num_1("")
        assert str(excinfo.value) == "The input 'loop' param is not valid."

    def test_check_min_num_2_non_digit_string(self):
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_min_num_2("abc")
        assert str(excinfo.value) == "The input 'threshold' param is not valid."

    def test_check_min_num_2_negative_valid(self):
        value = "-1"
        result = check_min_num_2(value)
        assert result == -1

    def test_check_min_num_2_negative_invalid(self):
        value = "-2"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_min_num_2(value)
        assert str(excinfo.value) == "-2 is not a valid value. Minimum value -1."

    def test_check_shapes_string_none(self):
        value = None
        result = check_shapes_string(value)
        assert result is None

    def test_check_shapes_string_legal(self):
        value = "1,3,224,224"
        result = check_shapes_string(value)
        assert result == value

    def test_check_shapes_string_illegal(self):
        value = "1,3,224,224,$"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_shapes_string(value)
        assert "is not a legal string" in str(excinfo.value)

    def test_check_dtypes_string_empty(self):
        value = ""
        result = check_dtypes_string(value)
        assert result == value

    def test_check_dtypes_string_legal(self):
        value = "float32;int64:float64"
        result = check_dtypes_string(value)
        assert result == value

    def test_check_dtypes_string_illegal(self):
        value = "float32;int64:float64@"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_dtypes_string(value)
        assert "dtypes string" in str(excinfo.value)

    def test_check_io_string_none(self):
        value = None
        result = check_io_string(value)
        assert result is None

    def test_check_io_string_legal(self):
        value = "input1,output2:input3;output4"
        result = check_io_string(value)
        assert result == value

    def test_check_io_string_illegal(self):
        value = "input1,output2:input3;output4$"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_io_string(value)
        assert "io string" in str(excinfo.value)

    def test_check_nodes_string_none(self):
        value = None
        result = check_nodes_string(value)
        assert result is None

    def test_check_nodes_string_legal(self):
        value = "node1,node2:node3/node4"
        result = check_nodes_string(value)
        assert result == value

    def test_check_nodes_string_illegal(self):
        value = "node1,node2:node3$node4"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_nodes_string(value)
        assert "nodes string" in str(excinfo.value)

    def test_check_single_node_string_none(self):
        value = None
        result = check_single_node_string(value)
        assert result is None

    def test_check_single_node_string_legal(self):
        value = "node_1:0/CPU:0.0"
        result = check_single_node_string(value)
        assert result == value

    def test_check_single_node_string_illegal(self):
        value = "node_1:0/CPU:0.0$"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_single_node_string(value)
        assert "single_node string" in str(excinfo.value)

    def test_check_normal_string_none(self):
        value = None
        result = check_normal_string(value)
        assert result is None

    def test_check_normal_string_legal(self):
        value = "This is a normal string with _ and -."
        result = check_normal_string(value)
        assert result == value

    def test_check_normal_string_illegal(self):
        value = "This string contains an illegal $ymbol"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_normal_string(value)
        assert "single_node string" in str(excinfo.value)

    def test_check_shapes_range_string_none(self):
        value = None
        result = check_shapes_range_string(value)
        assert result is None

    def test_check_shapes_range_string_legal(self):
        value = "1,2:3,4-5"
        result = check_shapes_range_string(value)
        assert result == value

    def test_check_shapes_range_string_illegal_character(self):
        value = "1,2:3,4-5$"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_shapes_range_string(value)
        assert "dym range string" in str(excinfo.value)

    def test_check_ints_string_none(self):
        value = None
        result = check_ints_string(value)
        assert result is None

    def test_check_ints_string_legal(self):
        value = "1,2,3,4,5"
        result = check_ints_string(value)
        assert result == value

    def test_check_ints_string_illegal(self):
        value = "1,2,three,4,5"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_ints_string(value)
        assert "ints string" in str(excinfo.value)

    def test_check_path_string_none(self):
        value = None
        result = check_path_string(value)
        assert result is None

    def test_check_path_string_illegal(self):
        value = "not/a/legal:path"
        with pytest.raises(ArgumentTypeError) as excinfo:
            check_path_string(value)
        assert f"\"{value}\" is not a legal string" in str(excinfo.value)


if __name__ == '__main__':
    unittest.main()
