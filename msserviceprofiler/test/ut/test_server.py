import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call

import numpy as np
from xmlrpc.server import SimpleXMLRPCServer

# Mock配置
class MockSettings:
    def __init__(self):
        self.mindie = "mock_mindie"
        self.target_field = ["field1", "field2"]

class MockSimulator:
    def __init__(self, mindie):
        self.mindie = mindie
        self.mindie_log = "mock_log.txt"
        self.process = MagicMock()

    def run(self, params):
        pass

    def check_success(self):
        return True

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
    from msserviceprofiler.modelevalstate.optimizer.server import (
        get_file,
        RequestHandler,
        RemoteScheduler,
        main
    )

    def setUp(self):
        # 创建临时目录和文件用于测试
        self.test_dir = Path("test_dir")
        self.test_dir.mkdir(exist_ok=True)
        self.test_file = self.test_dir / "test_file.txt"
        self.test_file.touch()
        self.test_sub_dir = self.test_dir / "sub_dir"
        self.test_sub_dir.mkdir(exist_ok=True)
        self.test_sub_file = self.test_sub_dir / "test_sub_file.txt"
        self.test_sub_file.touch()

    def tearDown(self):
        # 清理测试创建的临时目录和文件
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def test_get_file_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            get_file("nonexistent_path")

    def test_get_file_single_file(self):
        result = get_file(self.test_file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "test_file.txt")
        self.assertIsInstance(result[0][1], xmlrpc.client.Binary)

    def test_get_file_directory(self):
        result = get_file(self.test_dir)
        self.assertEqual(len(result), 2)
        self.assertIn(("test_file.txt", xmlrpc.client.Binary(b'')), result)
        self.assertIn(("sub_dir/test_sub_file.txt", xmlrpc.client.Binary(b'')), result)

    def test_get_file_with_parent_name(self):
        result = get_file(self.test_file, "parent")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "parent/test_file.txt")
        self.assertIsInstance(result[0][1], xmlrpc.client.Binary)

    def test_get_file_save_current_path(self):
        result = get_file(self.test_dir, save_current_path=True)
        self.assertEqual(len(result), 2)
        self.assertIn(("test_dir/test_file.txt", xmlrpc.client.Binary(b'')), result)
        self.assertIn(("test_dir/sub_dir/test_sub_file.txt", xmlrpc.client.Binary(b'')), result)


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


if __name__ == '__main__':
    unittest.main()