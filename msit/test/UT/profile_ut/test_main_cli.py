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
import argparse
from components.utils.parser import BaseCommand
from components.profile.msit_prof.main_cli import (
    ProfileCommand, AnalyzeCommand,
    check_application_string_legality,
    get_cmd_instance
)


class TestProfileCommand(unittest.TestCase):
    def test_check_application_string_legality(self):
        # 测试合法应用字符串
        legal_str = "legal_app_str123"
        self.assertEqual(check_application_string_legality(legal_str), legal_str)
    
    def test_check_application_string_illegal(self):
        # 测试非法应用字符串
        with self.assertRaises(argparse.ArgumentTypeError):
            check_application_string_legality("illegal@app_str!")

    @patch('argparse.ArgumentParser')
    def test_add_arguments(self, mock_parser):
        cmd = ProfileCommand("profile", "help_info")
        cmd.add_arguments(mock_parser)
        mock_parser.add_argument.assert_called()


class TestAnalyzeCommand(unittest.TestCase):
    @patch('argparse.ArgumentParser')
    def test_add_arguments(self, mock_parser):
        cmd = AnalyzeCommand("analyze", "help_info")
        cmd.add_arguments(mock_parser)
        mock_parser.add_argument.assert_called()

    def test_get_cmd_instance(self):
        instance = get_cmd_instance()
        self.assertIsInstance(instance, BaseCommand)
        self.assertEqual(instance.name, "profile")
        self.assertEqual(instance.help_info, "Provides a one-stop performance tuning and analysis tools.")