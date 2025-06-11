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


class TestMain(unittest.TestCase):
    @patch('xmlrpc.server.SimpleXMLRPCServer')
    def test_main_server_setup(self, mock_server):
        # 设置mock服务器
        mock_server_instance = MagicMock()
        mock_server.return_value.__enter__.return_value = mock_server_instance
        
        # 模拟KeyboardInterrupt来停止服务器
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

        # 测试main函数
        with self.assertRaises(SystemExit):
            main('localhost', 8000)

        # 验证服务器设置
        self.assertTrue(mock_server_instance.register_introspection_functions.called)
        self.assertTrue(mock_server_instance.register_function.called)
        self.assertTrue(mock_server_instance.register_instance.called)
        self.assertTrue(mock_server_instance.serve_forever.called)

    @patch('xmlrpc.server.SimpleXMLRPCServer')
    def test_main_server_functions_registration(self, mock_server):
        # 设置mock服务器
        mock_server_instance = MagicMock()
        mock_server.return_value.__enter__.return_value = mock_server_instance
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

        # 测试main函数
        with self.assertRaises(SystemExit):
            main('localhost', 8000)

        # 验证注册的具体函数
        register_function_calls = mock_server_instance.register_function.call_args_list
        self.assertEqual(len(register_function_calls), 2)  # 应该注册了get_file和remove_file
        self.assertEqual(register_function_calls[0][0][0], get_file)
        self.assertEqual(register_function_calls[1][0][0], optimizer_mock.remove_file)

class TestRequestHandler(unittest.TestCase):
    def test_rpc_paths(self):
        self.assertEqual(RequestHandler.rpc_paths, ('/RPC2',))

if __name__ == '__main__':
    unittest.main()