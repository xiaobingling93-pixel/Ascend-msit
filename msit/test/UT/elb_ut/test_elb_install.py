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
import unittest
from unittest import mock
from unittest.mock import patch

from components.expert_load_balancing.elb.__install__ import ExpertLoadBalanceInstall


class TestExpertLoadBalanceInstallCheck(unittest.TestCase):
    @patch('pkg_resources.working_set')
    def test_check_ok(self, mock_working_set):
        mock_pkg = unittest.mock.MagicMock()
        mock_pkg.key = "ortools"
        mock_working_set.__iter__.return_value = [mock_pkg]
        
        result = ExpertLoadBalanceInstall.check()
        self.assertEqual(result, "OK")

    @patch('pkg_resources.working_set')
    def test_check_missing_ortools(self, mock_working_set):
        mock_working_set.__iter__.return_value = []
        
        result = ExpertLoadBalanceInstall.check()
        self.assertEqual(result, "[error] ortools not installed. Please install ortools packages.")

    @patch('pkg_resources.working_set')
    def test_check_multiple_missing_packages(self, mock_working_set):
        mock_pkg1 = unittest.mock.MagicMock()
        mock_pkg1.key = "other_package"
        mock_working_set.__iter__.return_value = [mock_pkg1]
        
        result = ExpertLoadBalanceInstall.check()
        self.assertIn("ortools not installed", result)
        self.assertEqual(len(result.split("\n")), 1)  # 当前只有ortools一个检查项