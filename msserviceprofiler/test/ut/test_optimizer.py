import os
import psutil
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call

import numpy as np
import pandas as pd
from loguru import logger

class TestKillChildren(unittest.TestCase):
    @patch('psutil.Process')
    @patch('logger.error')
    def test_kill_children(self, mock_logger, mock_process):
        mock_child = MagicMock()
        mock_child.is_running.return_value = True
        mock_child.send_signal.return_value = None
        mock_child.wait.return_value = None
        mock_child.pid = 1234

        kill_children([mock_child])

        mock_child.send_signal.assert_called_once_with(9)
        mock_child.wait.assert_called_once_with(10)
        mock_logger.assert_not_called()

        # 测试进程仍在运行的情况
        mock_child.is_running.return_value = True
        kill_children([mock_child])
        mock_logger.assert_called_once_with("Failed to kill the 1234 process.")

class TestKillProcess(unittest.TestCase):
    @patch('psutil.process_iter')
    @patch('TestKillChildren.kill_children')
    def test_kill_process(self, mock_kill_children, mock_process_iter):
        mock_proc = MagicMock()
        mock_proc.info = {"name": "test_process"}
        mock_process_iter.return_value = [mock_proc]

        kill_process("test_process")

        mock_kill_children.assert_called()

class TestRemoveFile(unittest.TestCase):
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('shutil.rmtree')
    @patch('pathlib.Path.iterdir')
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.unlink')
    def test_remove_file(self, mock_unlink, mock_is_file, mock_iterdir, mock_rmtree, mock_exists):
        mock_exists.return_value = True
        mock_is_file.return_value = True

        remove_file("test_path")

        mock_unlink.assert_called_once()

        # 测试目录情况
        mock_is_file.return_value = False
        mock_iterdir.return_value = [MagicMock()]
        remove_file("test_path")
        mock_rmtree.assert_called_once()

class TestBackup(unittest.TestCase):
    @patch('shutil.copytree')
    @patch('shutil.copy')
    @patch('pathlib.Path.joinpath')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.mkdir')
    def test_backup(self, mock_mkdir, mock_is_file, mock_exists, mock_joinpath, mock_copy, mock_copytree):
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_joinpath.return_value = MagicMock()

        backup("target_path", "bak_path")

        mock_copy.assert_called_once()

        # 测试目录情况
        mock_is_file.return_value = False
        mock_copytree.assert_called_once()

class TestCloseFileFp(unittest.TestCase):
    @patch('os.close')
    @patch('builtins.hasattr')
    def test_close_file_fp(self, mock_hasattr, mock_os_close):
        mock_file_fp = MagicMock()
        mock_hasattr.return_value = True

        close_file_fp(mock_file_fp)

        mock_file_fp.close.assert_called_once()

        # 测试文件描述符情况
        mock_hasattr.return_value = False
        close_file_fp(1234)
        mock_os_close.assert_called_once_with(1234)

class TestClearingResidualProcess(unittest.TestCase):
    @patch('TestKillProcess.kill_process')
    @patch('MindieConfig')
    def test_clearing_residual_process(self, mock_mindie_config, mock_kill_process):
        mock_mindie_config.return_value.process_name = "test_process"

        clearing_residual_process()

        mock_kill_process.assert_called_once_with("test_process")


# Mock配置
class MockSettings:
    def __init__(self):
        self.mindie = "mock_mindie"
        self.target_field = ["field1", "field2"]

class MockBenchMarkConfig:
    def __init__(self):
        self.output_path = "mock_output"
        self.custom_collect_output_path = "mock_custom_output"
        self.work_path = "mock_work_path"
        self.command = "mock command"
        self.profile_input_path = "mock_profile_input"
        self.profile_output_path = "mock_profile_output"

class MockMindieConfig:
    def __init__(self):
        self.process_name = "mock_process"

# 替换原始导入
config_mock = MagicMock()
config_mock.settings = MockSettings()
config_mock.MindieConfig = MockMindieConfig
config_mock.AnalyzeTool = MagicMock()
config_mock.BenchMarkConfig = MockBenchMarkConfig
config_mock.PerformanceIndex = MagicMock()
config_mock.OptimizerConfigField = MagicMock()

with patch.dict('sys.modules', {
    'msserviceprofiler.modelevalstate.config.config': config_mock,
}):
    from msserviceprofiler.modelevalstate.optimizer.optimizer import (
        kill_children,
        kill_process,
        remove_file,
        backup,
        close_file_fp,
        BenchMark,
        ProfilerBenchmark
    )

class TestOptimizerUtils(unittest.TestCase):
    def setUp(self):
        # 创建临时测试目录
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.txt"
        with open(self.test_file, "w") as f:
            f.write("test content")

    def tearDown(self):
        # 清理测试目录
        shutil.rmtree(self.test_dir)

    @patch('psutil.Process')
    def test_kill_children(self, mock_process):
        # 创建mock子进程
        mock_child = MagicMock()
        mock_child.is_running.side_effect = [True, False]
        mock_child.pid = 12345

        # 测试成功杀死进程
        kill_children([mock_child])
        mock_child.send_signal.assert_called_once_with(9)
        mock_child.wait.assert_called_once_with(10)

        # 测试进程已经停止
        mock_child.reset_mock()
        mock_child.is_running.return_value = False
        kill_children([mock_child])
        mock_child.send_signal.assert_not_called()

    @patch('psutil.process_iter')
    @patch('psutil.Process')
    def test_kill_process(self, mock_process_class, mock_process_iter):
        # 设置mock进程
        mock_proc = MagicMock()
        mock_proc.info = {"name": "test_process", "pid": 12345}
        mock_process_iter.return_value = [mock_proc]
        
        # 设置子进程
        mock_child = MagicMock()
        mock_process_class.return_value.children.return_value = [mock_child]

        # 执行kill_process
        kill_process("test_process")
        mock_process_class.assert_called_once_with(12345)
        mock_process_class.return_value.children.assert_called_once_with(recursive=True)

    def test_remove_file(self):
        # 测试删除文件
        remove_file(self.test_file)
        self.assertFalse(self.test_file.exists())

        # 测试删除不存在的文件
        remove_file(self.test_dir / "nonexistent.txt")

        # 测试删除目录
        test_subdir = self.test_dir / "subdir"
        test_subdir.mkdir()
        (test_subdir / "test.txt").touch()
        remove_file(test_subdir)
        self.assertFalse(test_subdir.exists())

    def test_backup(self):
        # 创建源文件和备份目录
        bak_dir = self.test_dir / "backup"
        bak_dir.mkdir()
        
        # 测试文件备份
        backup(self.test_file, bak_dir, "TestClass")
        backup_file = bak_dir / "TestClass" / self.test_file.name
        self.assertTrue(backup_file.exists())
        with open(backup_file, "r") as f:
            self.assertEqual(f.read(), "test content")

        # 测试不存在的文件
        backup(self.test_dir / "nonexistent.txt", bak_dir)
        self.assertFalse((bak_dir / "nonexistent.txt").exists())

    def test_close_file_fp(self):
        # 测试关闭文件对象
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            close_file_fp(temp_file)
            self.assertTrue(temp_file.closed)

        # 测试关闭文件描述符
        fd = os.open(self.test_file, os.O_RDONLY)
        close_file_fp(fd)
        with self.assertRaises(OSError):
            os.close(fd)  # 应该抛出错误，因为文件描述符已关闭

class TestBenchMark(unittest.TestCase):
    def setUp(self):
        self.benchmark_config = MockBenchMarkConfig()
        self.benchmark = BenchMark(self.benchmark_config)
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('pandas.read_csv')
    def test_get_performance_index(self, mock_read_csv):
        # 模拟CSV数据
        mock_df = pd.DataFrame({
            'OutputGenerateSpeed': ['10.5'],
            'Returned': ['95.5%'],
            'FirstTokenTime': ['500 ms'],
            'GeneratedTokenSpeed': ['20.5 t/s'],
            'DecodeTime': ['100 ms']
        })
        mock_read_csv.return_value = mock_df

        # 模拟文件系统
        with patch('pathlib.Path.iterdir') as mock_iterdir:
            mock_iterdir.return_value = [
                MagicMock(name='result_common.csv'),
                MagicMock(name='result_perf.csv')
            ]
            
            # 获取性能指标
            result = self.benchmark.get_performance_index()
            self.assertIsNotNone(result)

    @patch('subprocess.Popen')
    def test_run(self, mock_popen):
        # 设置mock进程
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        # 创建测试参数
        test_params = (
            MagicMock(config_position="env", name="TEST_ENV", value="test_value"),
        )

        # 运行benchmark
        self.benchmark.run(test_params)
        
        # 验证环境变量设置
        self.assertEqual(os.environ.get("TEST_ENV"), "test_value")
        
        # 验证进程创建
        self.assertIsNotNone(self.benchmark.process)
        mock_popen.assert_called_once()

    def test_check_success(self):
        # 测试进程未启动
        self.assertFalse(self.benchmark.check_success())

        # 测试进程运行中
        self.benchmark.process = MagicMock()
        self.benchmark.process.poll.return_value = None
        self.assertFalse(self.benchmark.check_success())

        # 测试进程成功完成
        self.benchmark.process.poll.return_value = 0
        self.assertTrue(self.benchmark.check_success())

        # 测试进程失败
        self.benchmark.process.poll.return_value = 1
        with self.assertRaises(subprocess.SubprocessError):
            self.benchmark.check_success()

    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.remove_file')
    def test_prepare(self, mock_remove):
        self.benchmark.prepare()
        self.assertEqual(mock_remove.call_count, 2)
        mock_remove.assert_has_calls([
            call(Path('mock_output')),
            call(Path('mock_custom_output'))
        ])

class TestProfilerBenchmark(unittest.TestCase):
    def setUp(self):
        self.benchmark_config = MockBenchMarkConfig()
        self.profiler = ProfilerBenchmark(self.benchmark_config)

    def test_init(self):
        self.assertEqual(self.profiler.profiler_cmd[0], "python")
        self.assertEqual(self.profiler.profiler_cmd[1], "-m")
        self.assertEqual(self.profiler.profiler_cmd[2], "ms_service_profiler.parse")
        self.assertIn(f"--input-path={self.benchmark_config.profile_input_path}", self.profiler.profiler_cmd)
        self.assertIn(f"--output-path={self.benchmark_config.profile_output_path}", self.profiler.profiler_cmd)

    def test_extra_performance_index(self):
        with self.assertRaises(ValueError):
            self.profiler.extra_performance_index()

if __name__ == '__main__':
    unittest.main()
