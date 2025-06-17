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

import argparse
import unittest
from unittest.mock import patch, MagicMock

from msprechecker.__main__ import main, run_precheck, run_env_dump, run_compare


class TestMain(unittest.TestCase):
    def setUp(self):
        self.mock = MagicMock()

    def test_no_command_should_should_raise_system_exit(self):
        self.assertRaises(BaseException, main)

    @patch("argparse.ArgumentParser.parse_known_args", return_value=(argparse.Namespace(command=2), None))
    @patch("argparse.ArgumentParser.print_help")
    def test_invalid_command_should_print_help(self, mock_print_help, _):
        mock_print_help.return_value = "123"
        self.assertIsNone(main())
        mock_print_help.assert_called_once()

    def test_precheck_command_should_call_run_precheck(self):
        with patch("argparse.ArgumentParser.parse_known_args",
                   return_value=(argparse.Namespace(func=run_precheck), None)):
            self.assertIsNone(main())

        with patch("argparse.ArgumentParser.parse_known_args", 
                   return_value=(argparse.Namespace(func=self.mock), None)):
            self.assertIsNone(main())
            self.mock.assert_called_once()
    
    def test_dump_command_should_call_run_env_dump(self):
        with patch("argparse.ArgumentParser.parse_known_args",
                   return_value=(argparse.Namespace(func=run_env_dump), None)):
            self.assertIsNone(main())
    
    def test_compare_command_should_call_run_compare(self):
        with patch("argparse.ArgumentParser.parse_known_args",
                   return_value=(argparse.Namespace(func=run_compare), None)):
            self.assertIsNone(main())
