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
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from argparse import ArgumentParser
from components.expert_load_balancing.elb.__main__ import ExpertLoadBalanceCommmand, check_input_path_legality



class TestExpertLoadBalanceCommandHandle(unittest.TestCase):

    def setUp(self):
        help_info = "Large Language Model(llm) Debugger Tools."
        self.command = ExpertLoadBalanceCommmand("expert-load-balancing", help_info)
        self.args = MagicMock()
        self.original_environ = dict(os.environ)
        self.original_path = list(sys.path)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_environ)
        sys.path = self.original_path

    @patch('os.path.exists')
    @patch('os.stat')
    @patch('os.getuid')
    @patch('pwd.getpwuid')
    @patch('components.expert_load_balancing.elb.__main__.logger')
    def test_handle_success(self, mock_logger, mock_getpwuid, mock_getuid, mock_stat, mock_exists):
        os.environ["ASCEND_TOOLKIT_HOME"] = "/valid/path"
        mock_exists.side_effect = lambda x: True
        
        mock_stat.return_value.st_uid = 1000
        mock_stat.return_value.st_mode = 0o750  # 安全权限
        mock_getuid.return_value = 1000
        mock_getpwuid.return_value.pw_name = "testuser"

        with patch.dict('sys.modules', {'elb.eplb_runner': MagicMock()}):
            self.command.handle(self.args)
        
        mock_logger.info.assert_any_call("===================load balancing algorithm start====================")
        mock_logger.info.assert_any_call("===================load balancing algorithm end====================")

    @patch('os.path.exists')
    @patch('os.stat')
    @patch('os.getuid')
    @patch('pwd.getpwuid')
    @patch('components.expert_load_balancing.elb.__main__.logger')
    def test_handle_import_failure(self, mock_logger, mock_getpwuid, mock_getuid, mock_stat, mock_exists):
        os.environ["ASCEND_TOOLKIT_HOME"] = "/valid/path"
        mock_exists.return_value = True
    
        mock_stat.return_value.st_uid = 1000
        mock_stat.return_value.st_mode = 0o750
        mock_getuid.return_value = 1000
        mock_getpwuid.return_value.pw_name = "testuser"
        
        with self.assertRaises(Exception) as context:
            self.command.handle(self.args)
        
        self.assertEqual(str(context.exception), "Failed to import load_balancing module")


class TestExpertLoadBalanceCommmandAddArguments(unittest.TestCase):
    def setUp(self):
        help_info = "Large Language Model(llm) Debugger Tools."
        self.command = ExpertLoadBalanceCommmand("expert-load-balancing", help_info)
        self.parser = ArgumentParser()
        self.mock_parser = MagicMock()

    def test_add_arguments_all_required(self):
        """测试添加所有必选参数"""
        with patch.object(self.mock_parser, 'add_argument') as mock_add:
            self.command.add_arguments(self.mock_parser)

            mock_add.assert_any_call(
                '--info-csv-path', '-icp',
                dest="expert_popularity_csv_load_path",
                required=True,
                type=check_input_path_legality,
                help="Data input directory. Contains  CSV fileswhich might have been generated during prefill or decoder."
            )
            mock_add.assert_any_call(
                '--device-type', '-dt',
                dest="device_type",
                type=str,
                required=True,
                choices=['a2', 'a3'],
                help="device type. a2 代表适用于Atlas 800I A2推理服务器, a3 代表适用于Atlas 800I A3推理服务器。"
            )