import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from msserviceprofiler.modelevalstate.optimizer.server import get_file, RemoteScheduler, main

class TestGetFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("test content")
        
        # Create nested directory structure
        self.nested_dir = os.path.join(self.temp_dir, "nested")
        os.makedirs(self.nested_dir)
        self.nested_file = os.path.join(self.nested_dir, "nested.txt")
        with open(self.nested_file, "w") as f:
            f.write("nested content")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            get_file("non_existent_path")

    def test_single_file(self):
        result = get_file(self.test_file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "test.txt")

    def test_directory(self):
        result = get_file(self.temp_dir)
        self.assertEqual(len(result), 2)  # Should find both files

    def test_with_parent_name(self):
        result = get_file(self.test_file, parent_name="parent")
        self.assertEqual(result[0][0], "parent/test.txt")

    def test_save_current_path(self):
        result = get_file(self.nested_dir, save_current_path=True)
        self.assertTrue(any("nested/nested.txt" in file_info[0] for file_info in result))

class TestRemoteScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = RemoteScheduler()

    def test_init(self):
        self.assertIsNone(self.scheduler.simulator)

    @patch('msserviceprofiler.modelevalstate.optimizer.server.Simulator')
    def test_run_simulator(self, mock_simulator):
        params = np.array([1, 2, 3])
        self.scheduler.run_simulator(params)
        mock_simulator.assert_called_once()
        self.assertIsNotNone(self.scheduler.simulator)

    @patch('msserviceprofiler.modelevalstate.optimizer.server.time.sleep')
    def test_check_success(self, mock_sleep):
        # Test when simulator is None
        self.assertIsNone(self.scheduler.check_success())

        # Test successful case
        self.scheduler.simulator = Mock()
        self.scheduler.simulator.check_success.return_value = True
        self.assertTrue(self.scheduler.check_success())

        # Test failure case
        self.scheduler.simulator.check_success.return_value = False
        self.scheduler.simulator.mindie_log = "log_path"
        with self.assertRaises(Exception):
            self.scheduler.check_success()

    def test_stop_simulator(self):
        # Test when simulator is None
        self.scheduler.stop_simulator()

        # Test with simulator
        self.scheduler.simulator = Mock()
        self.scheduler.stop_simulator(del_log=True)
        self.scheduler.simulator.stop.assert_called_once_with(True)

    def test_process_poll(self):
        # Test when simulator is None
        self.assertIsNone(self.scheduler.process_poll())

        # Test with simulator
        self.scheduler.simulator = Mock()
        self.scheduler.simulator.process.poll.return_value = 0
        self.assertEqual(self.scheduler.process_poll(), 0)

class TestMain(unittest.TestCase):
    @patch('msserviceprofiler.modelevalstate.optimizer.server.SimpleXMLRPCServer')
    def test_main_server_setup(self, mock_server):
        # Mock server instance
        mock_server_instance = MagicMock()
        mock_server.return_value.__enter__.return_value = mock_server_instance

        # Test normal server start
        main('localhost', 8000)

        # Verify server setup
        mock_server_instance.register_introspection_functions.assert_called_once()
        mock_server_instance.register_function.assert_any_call(get_file)
        mock_server_instance.serve_forever.assert_called_once()

    @patch('msserviceprofiler.modelevalstate.optimizer.server.SimpleXMLRPCServer')
    def test_main_keyboard_interrupt(self, mock_server):
        # Mock server to raise KeyboardInterrupt
        mock_server_instance = MagicMock()
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()
        mock_server.return_value.__enter__.return_value = mock_server_instance

        # Test KeyboardInterrupt handling
        with self.assertRaises(SystemExit) as cm:
            main('localhost', 8000)
        self.assertEqual(cm.exception.code, 0)

if __name__ == '__main__':
    unittest.main()
