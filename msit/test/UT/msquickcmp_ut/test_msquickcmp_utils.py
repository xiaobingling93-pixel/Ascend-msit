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
import unittest
from unittest.mock import patch
from components.debug.compare.msquickcmp.common.utils import check_exec_cmd, check_dynamic_shape, \
    AccuracyCompareException


class TestCheckExecCmd(unittest.TestCase):

    @patch('components.debug.compare.msquickcmp.common.utils.check_exec_script_file')
    def test_valid_command_with_two_args(self, mock_check_exec_script_file):
        mock_check_exec_script_file.return_value = None

        command = "python script.py"
        result = check_exec_cmd(command)

        # assert right
        mock_check_exec_script_file.assert_called_with('script.py')
        self.assertTrue(result)

    @patch('components.debug.compare.msquickcmp.common.utils.check_exec_script_file')
    def test_valid_command_with_more_than_two_args(self, mock_check_exec_script_file):
        mock_check_exec_script_file.return_value = None
        command = "python script.py arg1 arg2"
        result = check_exec_cmd(command)

        # assert right
        mock_check_exec_script_file.assert_called_with('script.py')
        self.assertTrue(result)

    def test_invalid_Command(self):
        command = 'bash'
        with self.assertRaises(AccuracyCompareException):
            check_exec_cmd(command)


class TestCheckDynamicShape(unittest.TestCase):
    def test_no_dynamic_shape(self):
        shape = [1,2,3,4]
        result = check_dynamic_shape(shape)
        self.assertFalse(result, "The shape should not be dynamic")

    def test_dynamic_shape_with_none(self):
        shape = [1,2,3,None]
        result = check_dynamic_shape(shape)
        self.assertTrue(result, "The shape should be dynamic due to None")

    def test_dynamic_shape_with_string(self):
        shape = [1,2,'unknown',4]
        result = check_dynamic_shape(shape)
        self.assertTrue(result, "The shape should be dynamic due to a string")

    def test_empty_dynamic(self):
        shape = []
        result = check_dynamic_shape(shape)
        self.assertFalse(result, "An empty shape should be considered dynamic")

    def test_dynamic_shape_with_mixed_value(self):
        shape = [1,2,'dynamic',None]
        result = check_dynamic_shape(shape)
        self.assertTrue(result, "The shape should be dynamic due to None and string")