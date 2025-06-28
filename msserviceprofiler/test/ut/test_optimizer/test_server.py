# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call
import xmlrpc
from xmlrpc.server import SimpleXMLRPCServer
import numpy as np


class MockSettings:
    def __init__(self):
        self.mindie = "mock_mindie"
        self.target_field = ["field1", "field2"]


class MockSimulator:
    def __init__(self, mindie):
        self.mindie = mindie
        self.mindie_log = "mock_log.txt"
        self.process = MagicMock()

    @staticmethod
    def check_success():
        return True

    def run(self, params):
        pass

    def stop(self, del_log=True):
        pass


def mock_map_param_with_value(params, target_field):
    return [("param1", 1.0), ("param2", 2.0)]

# 替换原始导入
config_mock = MagicMock()
config_mock.settings = MockSettings()
config_mock.map_param_with_value = mock_map_param_with_value

optimizer_mock = MagicMock()
optimizer_mock.Simulator = MockSimulator
optimizer_mock.remove_file = MagicMock()

with patch.dict('sys.modules', {
    'msserviceprofiler.modelevalstate.config.config': config_mock,
    'msserviceprofiler.modelevalstate.optimizer.optimizer': optimizer_mock
}):
    from msserviceprofiler.modelevalstate.optimizer.server import RequestHandler, RemoteScheduler


    class TestRemoteScheduler(unittest.TestCase):
        def setUp(self):
            self.scheduler = RemoteScheduler()

        def test_run_simulator(self):
            params = np.array([1.0, 2.0])
            self.scheduler.run_simulator(params)
            self.assertIsNotNone(self.scheduler.simulator)

        @patch('time.sleep')
        def test_check_success(self, mock_sleep):
            # 测试simulator未初始化的情况
            self.assertIsNone(self.scheduler.check_success())

            # 测试正常运行的情况 - 立即成功
            self.scheduler.simulator = MockSimulator("mock_mindie")
            self.assertTrue(self.scheduler.check_success())
            mock_sleep.assert_not_called()

        @patch('time.sleep')
        def test_check_success_failure(self, mock_sleep):
            # 准备一个总是失败的simulator
            self.scheduler.simulator = MockSimulator("mock_mindie")
            self.scheduler.simulator.check_success = Mock(return_value=False)

            # 测试最大重试次数和失败异常
            with self.assertRaises(Exception) as context:
                self.scheduler.check_success()

            # 验证异常信息
            self.assertIn("Simulator run failed", str(context.exception))
            self.assertIn("mock_log.txt", str(context.exception))

            # 验证重试次数和等待时间
            self.assertEqual(self.scheduler.simulator.check_success.call_count, 10)
            self.assertEqual(mock_sleep.call_count, 10)
            mock_sleep.assert_has_calls([call(10)] * 9)

        def test_stop_simulator(self):
            # 测试simulator未初始化的情况
            self.scheduler.stop_simulator()  # 不应抛出异常

            # 测试正常停止的情况
            self.scheduler.simulator = MockSimulator("mock_mindie")
            self.scheduler.stop_simulator(del_log=True)

        def test_process_poll(self):
            # 测试simulator未初始化的情况
            self.assertIsNone(self.scheduler.process_poll())

            # 测试正常poll的情况
            self.scheduler.simulator = MockSimulator("mock_mindie")
            self.scheduler.simulator.process.poll.return_value = 0
            self.assertEqual(self.scheduler.process_poll(), 0)

        def test_run_simulator_with_empty_params(self):
            # 测试空参数
            params = np.array([])
            self.scheduler.run_simulator(params)
            self.assertIsNotNone(self.scheduler.simulator)


    class TestRequestHandler(unittest.TestCase):
        def test_rpc_paths(self):
            self.assertEqual(RequestHandler.rpc_paths, ('/RPC2',))
