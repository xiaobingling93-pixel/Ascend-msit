import unittest
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path
import xmlrpc.client
import numpy as np

from msserviceprofiler.modelevalstate.optimizer.server import get_file, RemoteScheduler, main

class TestGetFile(unittest.TestCase):
    @patch('pathlib.Path')
    def test_get_file_not_exists(self, mock_path):
        # 测试文件不存在的情况
        mock_path.return_value.exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            get_file("non_existent_path")

    @patch('pathlib.Path')
    @patch('builtins.open')
    def test_get_file_single_file(self, mock_open, mock_path):
        # 测试单个文件的情况
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_file.return_value = True
        mock_path.return_value.name = "test.txt"
        
        mock_file = MagicMock()
        mock_file.read.return_value = b"test content"
        mock_open.return_value.__enter__.return_value = mock_file

        result = get_file("test.txt")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "test.txt")
        self.assertIsInstance(result[0][1], xmlrpc.client.Binary)

    @patch('pathlib.Path')
    def test_get_file_directory(self, mock_path):
        # 测试目录的情况
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_file.return_value = False
        mock_path.return_value.name = "test_dir"
        
        mock_child = MagicMock()
        mock_child.exists.return_value = True
        mock_child.is_file.return_value = True
        mock_child.name = "child.txt"
        
        mock_path.return_value.iterdir.return_value = [mock_child]
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=b"test content")):
            result = get_file("test_dir", save_current_path=True)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][0], "test_dir/child.txt")

class TestRemoteScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = RemoteScheduler()

    @patch('msserviceprofiler.modelevalstate.optimizer.server.Simulator')
    def test_run_simulator(self, mock_simulator):
        # 测试运行模拟器
        params = np.array([1, 2, 3])
        self.scheduler.run_simulator(params)
        mock_simulator.assert_called_once()
        mock_simulator.return_value.run.assert_called_once()

    @patch('time.sleep')
    def test_check_success(self, mock_sleep):
        # 测试成功检查
        self.scheduler.simulator = Mock()
        self.scheduler.simulator.check_success.return_value = True
        
        result = self.scheduler.check_success()
        self.assertTrue(result)
        self.scheduler.simulator.check_success.assert_called_once()

    def test_check_success_no_simulator(self):
        # 测试无模拟器情况
        self.scheduler.simulator = None
        result = self.scheduler.check_success()
        self.assertIsNone(result)

    def test_stop_simulator(self):
        # 测试停止模拟器
        self.scheduler.simulator = Mock()
        self.scheduler.stop_simulator(True)
        self.scheduler.simulator.stop.assert_called_once_with(True)

    def test_process_poll(self):
        # 测试进程状态检查
        self.scheduler.simulator = Mock()
        self.scheduler.simulator.process.poll.return_value = 0
        result = self.scheduler.process_poll()
        self.assertEqual(result, 0)

    def test_process_poll_no_simulator(self):
        # 测试无模拟器情况下的进程状态检查
        self.scheduler.simulator = None
        result = self.scheduler.process_poll()
        self.assertIsNone(result)

class TestMain(unittest.TestCase):
    @patch('msserviceprofiler.modelevalstate.optimizer.server.SimpleXMLRPCServer')
    def test_main(self, mock_server):
        # 测试主函数
        mock_server_instance = MagicMock()
        mock_server.return_value.__enter__.return_value = mock_server_instance
        
        # 模拟KeyboardInterrupt以结束服务器
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()
        
        with self.assertRaises(SystemExit) as cm:
            main('localhost', 8000)
        
        self.assertEqual(cm.exception.code, 0)
        mock_server_instance.register_introspection_functions.assert_called_once()
        mock_server_instance.register_function.assert_called()
        mock_server_instance.register_instance.assert_called()

if __name__ == '__main__':
    unittest.main()
