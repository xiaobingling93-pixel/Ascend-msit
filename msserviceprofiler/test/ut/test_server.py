import unittest
from unittest.mock import patch, Mock, MagicMock, mock_open
from pathlib import Path
import xmlrpc.client
import numpy as np
import os

from msserviceprofiler.modelevalstate.optimizer.server import get_file, RemoteScheduler, main


class TestGetFile(unittest.TestCase):

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

    def test_check_success_with_none_simulator(self):
        """Test check_success when simulator is None"""
        scheduler = RemoteScheduler()
        result = scheduler.check_success()
        self.assertIsNone(result)

    @patch('time.sleep')  # 避免实际等待10秒
    def test_check_success_success_case(self, mock_sleep):
        """Test check_success when simulator check succeeds"""
        scheduler = RemoteScheduler()
        mock_simulator = Mock()
        mock_simulator.check_success.return_value = True
        scheduler.simulator = mock_simulator
        
        result = scheduler.check_success()
        
        self.assertTrue(result)
        mock_simulator.check_success.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('time.sleep')  # 避免实际等待10秒
    def test_check_success_failure_case(self, mock_sleep):
        """Test check_success when simulator check always fails"""
        scheduler = RemoteScheduler()
        mock_simulator = Mock()
        mock_simulator.check_success.return_value = False
        mock_simulator.mindie_log = "test_log.txt"
        scheduler.simulator = mock_simulator
        
        with self.assertRaises(Exception) as context:
            scheduler.check_success()
        
        self.assertIn("Simulator run failed", str(context.exception))
        self.assertEqual(mock_simulator.check_success.call_count, 10)
        self.assertEqual(mock_sleep.call_count, 9)  # 应该调用9次sleep

    @patch('your_module.settings')
    @patch('your_module.logger')
    @patch('your_module.Simulator')
    @patch('your_module.map_param_with_value')
    def test_run_simulator(self, mock_map_param_with_value, mock_simulator, mock_logger, mock_settings):
        # 创建 RemoteScheduler 实例
        remote_scheduler = RemoteScheduler()

        # 模拟参数
        params = np.array([1, 2, 3])

        # 模拟 map_param_with_value 的返回值
        mock_map_param_with_value.return_value = ('param1', 'param2', 'param3')

        # 调用 run_simulator 方法
        remote_scheduler.run_simulator(params)

        # 验证 Simulator 是否被正确实例化
        mock_simulator.assert_called_once_with(mock_settings.mindie)

        # 验证 map_param_with_value 是否被正确调用
        mock_map_param_with_value.assert_called_once_with(params, mock_settings.target_field)

        # 验证 logger.info 是否被正确调用
        mock_logger.info.assert_called_once_with("simulate run info ('param1', 'param2', 'param3')")

        # 验证 Simulator 的 run 方法是否被正确调用
        mock_simulator.return_value.run.assert_called_once_with(('param1', 'param2', 'param3'))

    @patch('your_module.settings')
    @patch('your_module.logger')
    @patch('your_module.Simulator')
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


if __name__ == '__main__':
    unittest.main()