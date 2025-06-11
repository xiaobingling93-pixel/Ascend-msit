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
        self.success_count = 0

    def run(self, params):
        pass

    def check_success(self):
        self.success_count += 1
        return self.success_count > 5

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

    def test_check_success(self):
        # 测试simulator未初始化的情况
        self.assertIsNone(self.scheduler.check_success())

        # 测试正常运行的情况
        self.scheduler.simulator = MockSimulator("mock_mindie")
        self.assertTrue(self.scheduler.check_success())

    def test_check_success_failure(self):
        # 测试检查失败的情况
        self.scheduler.simulator = MockSimulator("mock_mindie")
        self.scheduler.simulator.check_success = Mock(return_value=False)
        with self.assertRaises(Exception):
            self.scheduler.check_success()

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

    @patch('msserviceprofiler.modelevalstate.optimizer.server.settings')
    @patch('msserviceprofiler.modelevalstate.optimizer.server.logger')
    @patch('msserviceprofiler.modelevalstate.optimizer.server.Simulator')
    def test_run_simulator_with_empty_params(self, mock_simulator, mock_logger, mock_settings):
        # 创建 RemoteScheduler 实例
        remote_scheduler = RemoteScheduler()

        # 模拟空参数
        params = np.array([])

        # 模拟 map_param_with_value 的返回值
        mock_map_param_with_value.return_value = ()

        # 调用 run_simulator 方法
        remote_scheduler.run_simulator(params)

        # 验证 Simulator 是否被正确实例化
        mock_simulator.assert_called_once_with(mock_settings.mindie)

        # 验证 map_param_with_value 是否被正确调用
        mock_map_param_with_value.assert_called_once_with(params, mock_settings.target_field)

        # 验证 logger.info 是否被正确调用
        mock_logger.info.assert_called_once_with("simulate run info ()")

        # 验证 Simulator 的 run 方法是否被正确调用
        mock_simulator.return_value.run.assert_called_once_with(())


class TestRequestHandler(unittest.TestCase):
    def test_rpc_paths(self):
        self.assertEqual(RequestHandler.rpc_paths, ('/RPC2',))


from msserviceprofiler.modelevalstate.optimizer.server import main
class TestServer(unittest.TestCase):

    @patch('xmlrpc.server.SimpleXMLRPCServer')
    @patch('logging.Logger')
    def test_server_initialization(self, mock_logger, mock_server):
        # 模拟服务器对象
        mock_instance = mock_server.return_value
        mock_instance.register_function = MagicMock()
        mock_instance.register_instance = MagicMock()

        # 测试服务器初始化
        main('localhost', 8000)

        # 检查服务器是否绑定到正确的地址
        mock_server.assert_called_once_with(('localhost', 8000), allow_none=True, requestHandler=RequestHandler)

        # 检查是否注册了必要的函数和实例
        mock_instance.register_function.assert_any_call(get_file)
        mock_instance.register_function.assert_any_call(remove_file)
        mock_instance.register_instance.assert_called_once_with(RemoteScheduler())

        # 检查日志信息
        mock_logger.info.assert_called_once_with("server info. host:{}, port: {}", 'localhost', 8000)


if __name__ == '__main__':
    unittest.main()