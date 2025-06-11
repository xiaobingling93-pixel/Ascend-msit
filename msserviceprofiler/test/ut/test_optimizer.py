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


import argparse
from xmlrpc.client import ServerProxy

# Import the classes to test
from msserviceprofiler.modelevalstate.optimizer.optimizer import (
    BenchMark, ProfilerBenchmark, VllmBenchMark,
    Simulator, VllmSimulator, Scheduler, 
    ScheduleWithMultiMachine, PSOOptimizer,
    main, arg_parse
)
from msserviceprofiler.modelevalstate.config.config import (
    BenchMarkConfig, MindieConfig, PerformanceIndex,
    OptimizerConfigField, PsoOptions, DeployPolicy,
    BenchMarkPolicy, AnalyzeTool
)


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

class TestBenchMark(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.benchmark_config = BenchMarkConfig(
            output_path=self.temp_dir / "output",
            command="test_command",
            work_path=self.temp_dir
        )
        self.benchmark = BenchMark(self.benchmark_config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('os.environ', {})
    def test_run_with_env_params(self):
        with patch('subprocess.Popen') as mock_popen:
            params = (
                OptimizerConfigField(name="param1", value=10, config_position="env"),
                OptimizerConfigField(name="param2", value=20, config_position="other")
            )
            self.benchmark.run(params)
            self.assertEqual(os.environ["param1"], "10")
            self.assertNotIn("param2", os.environ)

    def test_get_performance_index_no_files(self):
        with self.assertRaises(ValueError):
            self.benchmark.get_performance_index()

    @patch('builtins.open')
    @patch('os.path.exists', return_value=True)
    def test_check_success(self, mock_exists, mock_open):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.tell.side_effect = [0, 100]
        mock_file.read.return_value = "test output"
        
        self.benchmark.process = MagicMock()
        self.benchmark.process.poll.return_value = 0
        
        result = self.benchmark.check_success(print_log=True)
        self.assertTrue(result)

class TestProfilerBenchmark(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.benchmark_config = BenchMarkConfig(
            output_path=self.temp_dir / "output",
            profile_input_path=self.temp_dir / "input",
            profile_output_path=self.temp_dir / "profile_output",
            command="test_command"
        )
        self.benchmark = ProfilerBenchmark(
            self.benchmark_config,
            analyze_tool=AnalyzeTool.profiler
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        config_file = self.temp_dir / "config.json"
        config_file.write_text('{"key": "value"}')
        
        self.mindie_config = MindieConfig(
            config_path=config_file,
            config_bak_path=self.temp_dir / "config_bak.json",
            command="test_command",
            process_name="test_process"
        )
        self.simulator = Simulator(self.mindie_config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('subprocess.Popen')
    def test_run(self, mock_popen):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        params = (
            OptimizerConfigField(name="param1", value=10, config_position="BackendConfig.key"),
            OptimizerConfigField(name="param2", value=20, config_position="env.OTHER")
        )
        
        self.simulator.run(params)
        
        # Verify config was updated
        config_content = self.simulator.mindie_config.config_path.read_text()
        self.assertIn('"key": 10', config_content)
        
        # Verify process was started
        mock_popen.assert_called_once()
        self.assertEqual(self.simulator.process, mock_process)

    @patch('builtins.open')
    @patch('os.path.exists', return_value=True)
    def test_check_success(self, mock_exists, mock_open):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.tell.side_effect = [0, 100]
        mock_file.read.return_value = "Daemon start success!"
        
        self.simulator.process = MagicMock()
        self.simulator.process.poll.return_value = None
        
        result = self.simulator.check_success(print_log=True)
        self.assertTrue(result)

class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock dependencies
        self.mock_simulator = MagicMock(spec=Simulator)
        self.mock_benchmark = MagicMock(spec=BenchMark)
        self.mock_data_storage = MagicMock()
        
        self.scheduler = Scheduler(
            simulator=self.mock_simulator,
            benchmark=self.mock_benchmark,
            data_storage=self.mock_data_storage
        )


class TestPSOOptimizer(unittest.TestCase):
    def setUp(self):
        self.mock_scheduler = MagicMock(spec=Scheduler)
        
        self.target_field = (
            OptimizerConfigField(name="param1", min=0, max=10),
            OptimizerConfigField(name="param2", min=0, max=10)
        )
        
        self.pso_options = PsoOptions()
        
        self.optimizer = PSOOptimizer(
            scheduler=self.mock_scheduler,
            n_particles=5,
            iters=2,
            pso_options=self.pso_options,
            target_field=self.target_field
        )

    def test_minimum_algorithm(self):
        perf_index = PerformanceIndex(
            generate_speed=100,
            time_to_first_token=0.1,
            time_per_output_token=0.2,
            success_rate=0.95
        )
        
        result = self.optimizer.minimum_algorithm(perf_index)
        self.assertIsInstance(result, float)


class TestMainFunction(unittest.TestCase):
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.PSOOptimizer')
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.Scheduler')
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.DataStorage')
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.ProfilerBenchmark')
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.Simulator')
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.settings')
    def test_main(self, mock_settings, mock_simulator, mock_benchmark, 
                 mock_data_storage, mock_scheduler, mock_pso):
        # Setup test args
        args = argparse.Namespace(
            deploy_policy=DeployPolicy.single.value,
            benchmark_policy=BenchMarkPolicy.benchmark.value,
            load_breakpoint=False,
            backup=False
        )
        
        # Setup mock returns
        mock_pso_instance = MagicMock()
        mock_pso.return_value = mock_pso_instance
        
        # Run main
        main(args)
        
        # Verify PSO was run
        mock_pso_instance.run.assert_called_once()


if __name__ == '__main__':
    unittest.main()