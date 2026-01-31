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
import argparse
from unittest.mock import patch

from components.tensor_view.ait_tensor_view.main_cli import (
    check_output_path_legality,
    parse_operations,
    get_args,
    TensorViewCommand,
    get_cmd_instance,
)


class TestTensorViewCommand(unittest.TestCase):
    def test_check_output_path_legality_valid(self):
        # Test with a valid path
        valid_path = "./valid_output.bin"
        self.assertEqual(check_output_path_legality(valid_path), valid_path)

    def test_check_output_path_legality_invalid(self):
        # Test with an invalid path
        invalid_path = "/path/with@illegal$char&ers!"
        with self.assertRaises(argparse.ArgumentTypeError):
            check_output_path_legality(invalid_path)

    def test_parse_operations_valid(self):
        # Test with valid operations
        valid_operations = "[0:10];(1,2,0)"
        ops = parse_operations(valid_operations)
        self.assertEqual(len(ops), 2)

    def test_parse_operations_invalid(self):
        # Test with invalid operations
        invalid_operations = "[0:10];invalid_op"
        with self.assertRaises(SyntaxError):
            parse_operations(invalid_operations)

    def test_get_args(self):
        # Test the get_args function
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            get_args()
            mock_parse_args.assert_called_once()

    def test_get_cmd_instance(self):
        # Test the get_cmd_instance function
        cmd_instance = get_cmd_instance()
        self.assertIsInstance(cmd_instance, TensorViewCommand)
        self.assertEqual(cmd_instance.name, "tensor-view")
        self.assertEqual(cmd_instance.help_info, "view / slice / permute / save the dumped tensor")


if __name__ == "__main__":
    unittest.main()