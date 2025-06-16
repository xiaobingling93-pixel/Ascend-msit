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

import unittest
import json
from unittest.mock import patch, MagicMock, call

from msprechecker.prechecker.hardware_capacity.network_checker import (
    NetworkChecker,
    CheckResult,
)


class TestNetworkChecker(unittest.TestCase):
    def setUp(self):
        self.checker = NetworkChecker()

    @patch('msprechecker.prechecker.hardware_capacity.network_checker.run_shell_command')
    def test_ping_servers_success(self, mock_run_shell):
        mock_run_shell.return_value = MagicMock(
            returncode=0,
            stdout="rtt min/avg/max/mdev = 1.234/2.345/3.456/0.123 ms"
        )
        results = self.checker.ping_servers(["192.168.1.1"])
        self.assertEqual(results["192.168.1.1"], 2.345)

    @patch('msprechecker.prechecker.hardware_capacity.network_checker.run_shell_command')
    def test_ping_servers_failure(self, mock_run_shell):
        mock_run_shell.return_value = MagicMock(
            returncode=1,
            stdout=""
        )
        results = self.checker.ping_servers(["192.168.1.1"])
        self.assertEqual(results["192.168.1.1"], "error")

    def test_filter_hostips_k8s(self):
        test_data = {
            "items": [
                {
                    "metadata": {
                        "namespace": "mindie-system",
                        "labels": {"app": "mindie-server"}
                    },
                    "status": {
                        "phase": "Running",
                        "hostIP": "192.168.1.1"
                    }
                },
                {
                    "metadata": {
                        "namespace": "other-system",
                        "labels": {"app": "mindie-server"}
                    },
                    "status": {
                        "phase": "Running",
                        "hostIP": "192.168.1.2"
                    }
                }
            ]
        }
        results = self.checker.filter_hostips_k8s(test_data)
        self.assertEqual(results, ["192.168.1.1"])

    @patch('msprechecker.prechecker.hardware_capacity.network_checker.parse_ranktable_file')
    @patch('msprechecker.prechecker.hardware_capacity.network_checker.get_interface_by_ip')
    @patch.object(NetworkChecker, 'ping_servers')
    def test_collect_env_with_ranktable(self, mock_ping, mock_get_ip, mock_parse):
        mock_parse.return_value = {
            "server_list": [
                {"server_id": "192.168.1.1"},
                {"server_id": "192.168.1.2"}
            ]
        }
        mock_get_ip.return_value = (None, None)
        mock_ping.return_value = {"192.168.1.1": 1.23, "192.168.1.2": "error"}
        
        result = self.checker.collect_env()
        self.assertEqual(result, {"192.168.1.1": 1.23, "192.168.1.2": "error"})

    @patch.dict("sys.modules", {"psutil": MagicMock()})
    @patch('msprechecker.prechecker.hardware_capacity.network_checker.parse_ranktable_file')
    @patch('msprechecker.prechecker.hardware_capacity.network_checker.run_shell_command')
    def test_collect_env_k8s_success(self, mock_run_shell, mock_parse):
        mock_parse.return_value = None
        mock_run_shell.return_value = MagicMock(
            stdout=json.dumps({
                "items": [{
                    "metadata": {
                        "namespace": "mindie-system",
                        "labels": {"app": "mindie-server"}
                    },
                    "status": {
                        "phase": "Running",
                        "hostIP": "192.168.1.1"
                    }
                }]
            })
        )
        
        with patch.object(NetworkChecker, 'ping_servers') as mock_ping:
            mock_ping.return_value = {"192.168.1.1": 1.23}
            result = self.checker.collect_env()
            self.assertEqual(result, {"192.168.1.1": 1.23})

    @patch('msprechecker.prechecker.hardware_capacity.network_checker.parse_ranktable_file')
    @patch('msprechecker.prechecker.hardware_capacity.network_checker.run_shell_command')
    def test_collect_env_k8s_failure(self, mock_run_shell, mock_parse):
        mock_parse.return_value = None
        mock_run_shell.return_value = None
        
        result = self.checker.collect_env()
        self.assertIsNone(result)

    @patch('msprechecker.prechecker.hardware_capacity.network_checker.show_check_result')
    def test_do_precheck_success(self, mock_show):
        test_envs = {
            "192.168.1.1": 10.0,
            "192.168.1.2": 35.0,
            "192.168.1.3": "error"
        }
        self.checker.do_precheck(test_envs)
        
        # Verify calls were made correctly
        calls = [
            call(
                "hardware",
                "network_checker",
                CheckResult.ERROR,
                action="检查本机到服务器 192.168.1.3 的连接状态",
                reason="本机到对端卡的 ping 结果存在失败",
            ),
            call(
                "hardware",
                "network_checker",
                CheckResult.ERROR,
                action="检查本机到服务器 192.168.1.2 的连接状态",
                reason="本机到对端卡的 ping 时间为 35.0 ms 超过平均时间50%",
            ),
            call(
                "hardware",
                "network_checker 本机到服务器192.168.1.1的时间为10.0 ms",
                CheckResult.OK,
            )
        ]
        mock_show.assert_has_calls(calls, any_order=True)

    @patch('msprechecker.prechecker.hardware_capacity.network_checker.show_check_result')
    def test_do_precheck_no_valid_results(self, _):
        test_envs = {
            "192.168.1.1": "error",
            "192.168.1.2": "error"
        }
        self.assertIsNone(self.checker.do_precheck(test_envs))

    @patch('msprechecker.prechecker.hardware_capacity.network_checker.show_check_result')
    def test_do_precheck_empty_input(self, mock_show):
        self.checker.do_precheck({})
        mock_show.assert_not_called()
