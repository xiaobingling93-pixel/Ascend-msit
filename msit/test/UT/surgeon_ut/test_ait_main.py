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
import unittest
from unittest.mock import patch, MagicMock
from auto_optimizer.ait_main import (
    ListCommand,
    EvaluateCommand,
    OptimizeCommand
)


class TestSurgeonCommands(unittest.TestCase):

    def test_list_command_handle_given_no_args_when_called_then_list_knowledges_called(self):
        with patch('auto_optimizer.ait_main.list_knowledges') as mock_list_knowledges:
            cmd = ListCommand("list", "List available Knowledges")
            cmd.handle(None)
            mock_list_knowledges.assert_called_once()

    def test_evaluate_command_handle_given_valid_path_when_called_then_cli_eva_called(self):
        args = MagicMock()
        args.path = 'valid_path'
        args.knowledges = 'knowledge1,knowledge2'
        args.recursive = False
        args.verbose = False
        args.processes = 1

        with patch('auto_optimizer.ait_main.check_input_path', return_value=True), \
             patch('auto_optimizer.ait_main.pathlib.Path'), \
             patch('auto_optimizer.ait_main.GraphOptimizer'), \
             patch('auto_optimizer.ait_main.cli_eva') as mock_cli_eva:
            cmd = EvaluateCommand("evaluate", "Evaluate model matching specified knowledges", alias_name="eva")
            cmd.handle(args)
            mock_cli_eva.assert_called_once()

    def test_evaluate_command_handle_given_invalid_path_when_called_then_cli_eva_not_called(self):
        args = MagicMock()
        args.path = 'invalid_path'
        args.knowledges = 'knowledge1,knowledge2'
        args.recursive = False
        args.verbose = False
        args.processes = 1

        with patch('auto_optimizer.ait_main.check_input_path', return_value=False), \
             patch('auto_optimizer.ait_main.cli_eva') as mock_cli_eva:
            cmd = EvaluateCommand("evaluate", "Evaluate model matching specified knowledges", alias_name="eva")
            cmd.handle(args)
            mock_cli_eva.assert_not_called()

    def test_optimize_command_handle_given_invalid_paths_when_called_then_optimize_onnx_not_called(self):
        args = MagicMock()
        args.input_model = 'invalid_input_path'
        args.output_model = 'invalid_output_path'
        args.knowledges = 'knowledge1,knowledge2'
        args.infer_test = False
        args.big_kernel = False
        args.attention_start_node = ''
        args.attention_end_node = ''
        args.device = 0
        args.loop = 100
        args.threshold = 0
        args.input_shape = None
        args.input_shape_range = None
        args.dynamic_shape = None
        args.output_size = None

        with patch('auto_optimizer.ait_main.check_input_path', return_value=False), \
             patch('auto_optimizer.ait_main.check_output_model_path', return_value=False), \
             patch('auto_optimizer.ait_main.optimize_onnx') as mock_optimize_onnx:
            cmd = OptimizeCommand("optimize", "Optimize model with specified knowledges", alias_name="opt")
            cmd.handle(args)
            mock_optimize_onnx.assert_not_called()


if __name__ == '__main__':
    unittest.main()
