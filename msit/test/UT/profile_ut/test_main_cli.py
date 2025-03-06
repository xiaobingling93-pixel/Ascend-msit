# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import argparse
from components.profile.msprof.ait_prof.main_cli import (
    ProfileCommand,
    check_output_path_legality,
    check_application_string_legality,
    get_cmd_instance
)


class TestProfileCommand(unittest.TestCase):
    def test_check_output_path_legality(self):
        # 测试合法路径
        legal_path = "/path/to/legal/output"
        self.assertEqual(check_output_path_legality(legal_path), legal_path)
    
    def test_check_output_path_illegal(self):
        # 测试非法路径
        with self.assertRaises(argparse.ArgumentTypeError):
            check_output_path_legality("/ille@gal/path/output")
        
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

    def test_get_cmd_instance(self):
        instance = get_cmd_instance()
        self.assertIsInstance(instance, ProfileCommand)
        self.assertEqual(instance.name, "profile")
        self.assertEqual(instance.help_info, "get profiling data of a given programma")

if __name__ == '__main__':
    unittest.main()