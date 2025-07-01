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

import unittest
import tempfile
import shutil
import os
import argparse
import subprocess
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from xmlrpc.client import ServerProxy
import numpy as np
import pandas as pd
import pytest

from msserviceprofiler.modelevalstate.optimizer.optimizer import (
    BenchMark, ProfilerBenchmark, VllmBenchMark, Scheduler,
    ScheduleWithMultiMachine, PSOOptimizer,
    main, arg_parse, remove_file, backup,
    close_file_fp,
)
from msserviceprofiler.modelevalstate.optimizer.simulator import Simulator, VllmSimulator
from msserviceprofiler.modelevalstate.optimizer.utils import kill_process, kill_children
from msserviceprofiler.modelevalstate.config.config import (
    BenchMarkConfig, MindieConfig, PerformanceIndex, clearing_residual_process,
    OptimizerConfigField, PsoOptions, DeployPolicy,
    BenchMarkPolicy, AnalyzeTool, settings, default_support_field
)


class TestProfilerBenchmark(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.benchmark_config = BenchMarkConfig(
            output_path=self.temp_dir / "output",
            profile_input_path=self.temp_dir / "input",
            profile_output_path=self.temp_dir / "profile_output",
            command="test_command",
            work_path=self.temp_dir,
            custom_collect_output_path=self.temp_dir / "collect_output"
        )
        self.benchmark = ProfilerBenchmark(
            self.benchmark_config,
            analyze_tool=AnalyzeTool.profiler
        )

    def tearDown(self):
        if hasattr(self.benchmark, 'profiler_process') and self.benchmark.profiler_process:
            self.benchmark.profiler_process.kill()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        self.assertEqual(self.benchmark.analyze_tool, AnalyzeTool.profiler)
        self.assertIsNotNone(self.benchmark.profiler_cmd)
        self.assertIsNone(self.benchmark.profiler_log)
        self.assertIsNone(self.benchmark.profiler_log_fp)
        self.assertEqual(self.benchmark.profiler_log_offset, 0)
        self.assertIsNone(self.benchmark.profiler_process)

    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.remove_file')
    def test_prepare(self, mock_remove_file):
        self.benchmark.prepare()
        
        args_list = [args[0] for args, _ in mock_remove_file.call_args_list]
        self.assertIn(Path(self.benchmark_config.profile_input_path), args_list)
        self.assertIn(Path(self.benchmark_config.profile_output_path), args_list)

    @patch('subprocess.Popen')
    @patch('tempfile.mkstemp')
    def test_start_profiler(self, mock_mkstemp, mock_popen):
        mock_fd = 123
        mock_log_path = str(self.temp_dir / "mock_log")
        mock_mkstemp.return_value = (mock_fd, mock_log_path)
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        self.benchmark.start_profiler()

        self.assertEqual(self.benchmark.profiler_log, mock_log_path)
        self.assertEqual(self.benchmark.profiler_log_fp, mock_fd)
        self.assertEqual(self.benchmark.profiler_process, mock_process)
        mock_popen.assert_called_once()

    def test_check_profiler(self):
        self.benchmark.profiler_process = MagicMock()
        self.benchmark.process = MagicMock()

        self.benchmark.profiler_process.poll.return_value = None
        self.assertFalse(self.benchmark.check_profiler())

        self.benchmark.profiler_process.poll.return_value = 0
        self.assertTrue(self.benchmark.check_profiler())

        self.benchmark.profiler_process.poll.return_value = 1
        self.benchmark.process.returncode = 1
        with self.assertRaises(subprocess.SubprocessError):
            self.benchmark.check_profiler()

    def test_extra_performance_index(self):
        mock_analyze_tool = MagicMock()
        mock_analyze_tool.return_value = (10.0, 1.0, 0.1, 1.0)

        with patch.dict('msserviceprofiler.modelevalstate.optimizer.optimizer._analyze_mapping', 
                    {self.benchmark.analyze_tool: mock_analyze_tool}):
            
            result = self.benchmark.extra_performance_index("dummy_path", "collect_path")

            self.assertEqual(result.generate_speed, 10.0)
            self.assertEqual(result.time_to_first_token, 1.0)
            self.assertEqual(result.time_per_output_token, 0.1)
            self.assertEqual(result.success_rate, 1.0)

            mock_analyze_tool.assert_called_once_with("dummy_path", "collect_path")

    def test_analyze_tool_not_found(self):
        with patch.dict('msserviceprofiler.modelevalstate.optimizer.optimizer._analyze_mapping', {}, clear=True):
            with self.assertRaises(ValueError) as context:
                self.benchmark.extra_performance_index("dummy_path")
            
            self.assertIn(f"Analyze tool not found: {self.benchmark.analyze_tool}", 
                         str(context.exception))

    def test_extra_performance_index_with_single_value(self):
        mock_analyze_tool = MagicMock()
        mock_analyze_tool.return_value = 10.0
        
        with patch.dict('msserviceprofiler.modelevalstate.optimizer.optimizer._analyze_mapping', 
                    {self.benchmark.analyze_tool: mock_analyze_tool}):
            result = self.benchmark.extra_performance_index("dummy_path")
            
            self.assertEqual(result.generate_speed, 10.0)
            self.assertIsNone(result.time_to_first_token)
            self.assertIsNone(result.time_per_output_token)
            self.assertIsNone(result.success_rate)

    def test_extra_performance_index_with_1_tuple(self):
        mock_analyze_tool = MagicMock()
        mock_analyze_tool.return_value = (10.0,)
        
        with patch.dict('msserviceprofiler.modelevalstate.optimizer.optimizer._analyze_mapping', 
                    {self.benchmark.analyze_tool: mock_analyze_tool}):
            result = self.benchmark.extra_performance_index("dummy_path")
            
            self.assertEqual(result.generate_speed, 10.0)
            self.assertIsNone(result.time_to_first_token)
            self.assertIsNone(result.time_per_output_token)
            self.assertIsNone(result.success_rate)

    def test_extra_performance_index_with_2_tuple(self):
        mock_analyze_tool = MagicMock()
        mock_analyze_tool.return_value = (10.0, 1.0)
        
        with patch.dict('msserviceprofiler.modelevalstate.optimizer.optimizer._analyze_mapping', 
                    {self.benchmark.analyze_tool: mock_analyze_tool}):
            result = self.benchmark.extra_performance_index("dummy_path")
            
            self.assertEqual(result.generate_speed, 10.0)
            self.assertEqual(result.time_to_first_token, 1.0)
            self.assertIsNone(result.time_per_output_token)
            self.assertIsNone(result.success_rate)

    def test_extra_performance_index_with_3_tuple(self):
        mock_analyze_tool = MagicMock()
        mock_analyze_tool.return_value = (10.0, 1.0, 0.1)
        
        with patch.dict('msserviceprofiler.modelevalstate.optimizer.optimizer._analyze_mapping', 
                    {self.benchmark.analyze_tool: mock_analyze_tool}):
            result = self.benchmark.extra_performance_index("dummy_path")
            
            self.assertEqual(result.generate_speed, 10.0)
            self.assertEqual(result.time_to_first_token, 1.0)
            self.assertEqual(result.time_per_output_token, 0.1)
            self.assertIsNone(result.success_rate)

    def test_extra_performance_index_with_invalid_length(self):
        mock_analyze_tool = MagicMock()
        mock_analyze_tool.return_value = (10.0, 1.0, 0.1, 1.0, "extra")
        
        with patch.dict('msserviceprofiler.modelevalstate.optimizer.optimizer._analyze_mapping', 
                    {self.benchmark.analyze_tool: mock_analyze_tool}):
            with self.assertRaises(ValueError) as context:
                self.benchmark.extra_performance_index("dummy_path")
            self.assertIn("Not Support", str(context.exception))

    def test_backup_calls(self):
        # 设置备份路径
        self.benchmark.bak_path = self.temp_dir / "backup"
        os.makedirs(self.benchmark.bak_path, exist_ok=True)
        
        # 创建将被备份的模拟文件
        Path(self.benchmark_config.profile_input_path).touch()
        Path(self.benchmark_config.profile_output_path).touch()
        
        # 设置profiler_log路径并创建文件
        self.benchmark.profiler_log = str(self.temp_dir / "profiler.log")
        Path(self.benchmark.profiler_log).touch()
        
        # Mock父类的backup方法
        with patch.object(BenchMark, 'backup') as mock_super_backup:
            # Mock实际的backup函数(假设它在模块中定义)
            with patch('msserviceprofiler.modelevalstate.optimizer.optimizer.backup') as mock_backup_func:
                self.benchmark.backup(del_log=False)
                self.assertEqual(mock_backup_func.call_count, 3)

                calls = mock_backup_func.call_args_list
                self.assertEqual(calls[0][0], 
                            (self.benchmark_config.profile_input_path,
                                self.benchmark.bak_path,
                                self.benchmark.__class__.__name__))

                self.assertEqual(calls[1][0], 
                            (self.benchmark_config.profile_output_path,
                                self.benchmark.bak_path,
                                self.benchmark.__class__.__name__))

                self.assertEqual(calls[2][0], 
                            (self.benchmark.profiler_log,
                                self.benchmark.bak_path,
                                self.benchmark.__class__.__name__))

    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.Path')  # 如果需要mock Path对象
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.close_file_fp')  # mock自定义关闭文件函数
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.remove_file')  # mock自定义删除文件函数
    def test_stop(self, mock_remove_file, mock_close_file, mock_path):
        self.benchmark.run_log_fp = 123  # 模拟文件描述符
        self.benchmark.run_log = str(self.temp_dir / "mock_log")  # 模拟日志路径
        self.benchmark.process = MagicMock()  # 模拟进程对象
        self.benchmark.process.poll.return_value = None  # 设置进程正在运行

        self.benchmark.stop()

        self.assertEqual(mock_close_file.call_count, 2)
        self.assertEqual(mock_remove_file.call_count, 2)
        self.benchmark.process.kill.assert_called_once()

    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.close_file_fp')
    @patch('msserviceprofiler.modelevalstate.optimizer.optimizer.remove_file')
    def test_stop_without_deleting_log(self, mock_remove_file, mock_close_file):
        self.benchmark.run_log_fp = 123
        self.benchmark.process = MagicMock()
        self.benchmark.process.poll.return_value = None
        
        self.benchmark.stop(del_log=False)
        
        self.assertEqual(mock_close_file.call_count, 2)
        mock_remove_file.assert_not_called()  # 验证没有调用删除日志
        self.benchmark.process.kill.assert_called_once()


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

    def test_init_with_missing_config(self):
        with self.assertRaises(FileNotFoundError):
            Simulator(MindieConfig(
                config_path=Path("/nonexistent/config.json"),
                config_bak_path=self.temp_dir / "config_bak.json",
                command="test_command",
                process_name="test_process"
            ))

    def test_init_creates_backup_config(self):
        backup_path = self.temp_dir / "config_bak.json"
        backup_path.unlink()  # Remove backup file
        Simulator(self.mindie_config)  # Should recreate backup
        self.assertTrue(backup_path.exists())

    @patch('builtins.open')
    @patch('os.path.exists', return_value=True)
    def test_check_success(self, mock_exists, mock_open_func):
        mock_file = MagicMock()
        mock_open_func.return_value.__enter__.return_value = mock_file
        mock_file.tell.side_effect = [0, 100]
        mock_file.read.return_value = "Daemon start success!"
        
        self.simulator.process = MagicMock()
        self.simulator.process.poll.return_value = None
        
        result = self.simulator.check_success(print_log=True)
        self.assertTrue(result)

    def test_check_success_failure(self):
        with patch('builtins.open', mock_open(read_data="Failure message")):
            self.simulator.process = MagicMock()
            self.simulator.process.poll.return_value = 1
            
            with self.assertRaises(subprocess.SubprocessError):
                self.simulator.check_success()

    def test_set_config(self):
        test_config = {"a": 1, "b": {"c": 2}}
        Simulator.set_config(test_config, "b.c", 10)
        self.assertEqual(test_config["b"]["c"], 10)

    @patch('psutil.process_iter')
    def test_check_env(self, mock_process_iter):
        # Setup mock process
        mock_proc = MagicMock()
        mock_proc.info = {"name": "test_process", "pid": 123}
        mock_process_iter.return_value = [mock_proc]
        
        # Mock kill_process
        with patch('msserviceprofiler.modelevalstate.optimizer.utils.kill_process') as mock_kill:
            self.simulator.check_env()
            mock_kill.assert_called_once_with("test_process")

    @patch.object(Simulator, 'update_config')
    @patch.object(Simulator, 'check_env')
    @patch.object(Simulator, 'start_server')
    def test_run(self, mock_start, mock_check, mock_update):
        params = (
            OptimizerConfigField(name="param1", value=10, config_position="BackendConfig.key"),
        )
        self.simulator.run(params)
        
        mock_update.assert_called_once_with(params)
        mock_check.assert_called_once()
        mock_start.assert_called_once_with(params)

    @patch('psutil.Process')
    @patch('msserviceprofiler.modelevalstate.optimizer.utils.kill_process')
    @patch('msserviceprofiler.modelevalstate.optimizer.utils.kill_children')
    def test_stop(self, mock_kill_children, mock_kill_process, mock_process):
        # Setup running process
        self.simulator.process = MagicMock()
        self.simulator.process.pid = 123
        self.simulator.process.poll.return_value = None
        
        # Setup process children
        mock_proc = MagicMock()
        mock_process.return_value.children.return_value = [MagicMock()]
        
        self.simulator.stop()
        
        self.simulator.process.kill.assert_called_once()
        mock_kill_process.assert_called_once_with("test_process")


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


# 测试kill_children函数
class TestKillChildren:
    @patch('psutil.Process')
    def test_kill_children_success(self, mock_process):
        # 创建一个运行中的模拟子进程
        mock_child = MagicMock()
        mock_child.is_running.return_value = True
        mock_child.pid = 1234
        
        # 测试正常终止流程
        kill_children([mock_child])
        
        mock_child.send_signal.assert_called_with(9)
        mock_child.wait.assert_called_with(10)
    
    @patch('psutil.Process')
    def test_kill_children_already_stopped(self, mock_process):
        # 创建一个已停止的模拟子进程
        mock_child = MagicMock()
        mock_child.is_running.return_value = False
        
        kill_children([mock_child])
        
        mock_child.send_signal.assert_not_called()


# 测试 remove_file 函数
class TestRemoveFile:
    """测试 remove_file 函数的各种场景"""
    @classmethod
    @pytest.fixture
    def setup_test_files(cls, tmp_path):
        """创建测试用的临时文件和目录结构"""
        # 创建测试文件
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test content")
        
        # 创建测试目录
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        # 在目录中创建文件
        (test_dir / "nested_file.txt").write_text("nested content")
        
        # 创建嵌套目录
        nested_dir = test_dir / "nested_dir"
        nested_dir.mkdir()
        (nested_dir / "deep_file.txt").write_text("deep content")
        
        return tmp_path

    @classmethod
    def test_remove_single_file(cls, setup_test_files):
        """测试删除单个文件"""
        test_file = setup_test_files / "test_file.txt"
        assert test_file.exists()
        
        remove_file(test_file)
        assert not test_file.exists()

    @classmethod
    def test_remove_nonexistent_path(cls, setup_test_files):
        """测试删除不存在的路径"""
        non_existent = setup_test_files / "does_not_exist"
        assert not non_existent.exists()
        
        # 应该不会抛出异常
        remove_file(non_existent)

    @classmethod
    def test_remove_empty_path(cls):
        """测试传入空路径"""
        # 应该不会抛出异常
        remove_file(None)
        remove_file("")

    @classmethod
    def test_remove_string_path(cls, setup_test_files):
        """测试传入字符串路径"""
        test_file = setup_test_files / "test_file.txt"
        assert test_file.exists()
        
        remove_file(str(test_file))
        assert not test_file.exists()

    @classmethod
    @patch('shutil.rmtree')
    def test_remove_dir_failure(cls, mock_rmtree, setup_test_files):
        """测试删除目录时出现异常的情况"""
        test_dir = setup_test_files / "test_dir"
        
        # 模拟 rmtree 抛出异常
        mock_rmtree.side_effect = OSError("模拟删除目录失败")
        
        # 应该递归调用 remove_file
        remove_file(test_dir)
        assert mock_rmtree.call_count >= 1


class TestBackup:
    """测试 backup 函数的各种场景"""
    @classmethod
    @pytest.fixture
    def setup_test_dirs(cls, tmp_path):
        """创建测试用的临时目录结构"""
        # 源目录结构
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")
        (src_dir / "file2.txt").write_text("content2")
        sub_dir = src_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file3.txt").write_text("content3")

        # 备份目录
        bak_dir = tmp_path / "backup"
        bak_dir.mkdir()

        return src_dir, bak_dir

    @classmethod
    def test_backup_file(cls, setup_test_dirs):
        """测试备份单个文件"""
        src, bak = setup_test_dirs
        src_file = src / "file1.txt"
        
        backup(src_file, bak)
        
        bak_file = bak / "file1.txt"

    @classmethod
    def test_backup_directory(cls, setup_test_dirs):
        """测试备份整个目录"""
        src, bak = setup_test_dirs
        
        backup(src, bak)
        
        bak_subdir = bak / src.name / "subdir"

    @classmethod
    def test_backup_with_class_name(cls, setup_test_dirs):
        """测试带class_name参数的备份"""
        src, bak = setup_test_dirs
        src_file = src / "file1.txt"
        
        backup(src_file, bak, "test_class")
        
        bak_file = bak / "test_class" / "file1.txt"
        assert bak_file.exists()

    @classmethod
    def test_backup_existing_file_no_overwrite(cls, setup_test_dirs):
        """测试已存在文件不重复备份"""
        src, bak = setup_test_dirs
        src_file = src / "file1.txt"
        
        # 第一次备份
        backup(src_file, bak)
        # 第二次备份
        backup(src_file, bak)
        
        bak_file = bak / "file1.txt"

    @classmethod
    def test_backup_nonexistent_source(cls, setup_test_dirs):
        """测试源不存在的情况"""
        src, bak = setup_test_dirs
        non_existent = src / "not_exists.txt"
        
        backup(non_existent, bak)  # 不应报错
        assert not (bak / "not_exists.txt").exists()

    @classmethod
    def test_backup_empty_parameters(cls):
        """测试空参数"""
        backup(None, None)  # 不应报错
        backup("", "")  # 不应报错

    @classmethod
    def test_backup_existing_dir_with_class_name(cls, setup_test_dirs):
        """测试目标目录已存在且带class_name的情况"""
        src, bak = setup_test_dirs
        class_name = "test_class"
        
        # 先创建目标目录
        (bak / class_name).mkdir()
        
        # 备份文件
        backup(src / "file1.txt", bak, class_name)


class TestKillProcess(unittest.TestCase):
    @patch('psutil.process_iter')
    def test_kill_process_no_match(self, mock_process_iter):
        mock_proc = MagicMock()
        mock_proc.info = {"name": "other_process", "pid": 123}
        mock_process_iter.return_value = [mock_proc]

        kill_process("target_process")
        mock_proc.kill.assert_not_called()

    @patch('psutil.process_iter')
    def test_kill_process_no_info_attribute(self, mock_process_iter):
        mock_proc = MagicMock()
        del mock_proc.info
        mock_process_iter.return_value = [mock_proc]

        kill_process("target_process")
        mock_proc.kill.assert_not_called()

    @patch('msserviceprofiler.modelevalstate.optimizer.utils.psutil.process_iter')
    @patch('msserviceprofiler.modelevalstate.optimizer.utils.psutil.Process')
    @patch('msserviceprofiler.modelevalstate.optimizer.utils.kill_children')
    def test_kill_process_with_match(self, mock_kill_children, mock_process_class, mock_process_iter):
        mock_proc = MagicMock()
        mock_proc.info = {"name": "target_process", "pid": 123}
        mock_process_iter.return_value = [mock_proc]

        mock_child1 = MagicMock()
        mock_child2 = MagicMock()
        mock_process_instance = MagicMock()
        mock_process_instance.children.return_value = [mock_child1, mock_child2]
        mock_process_class.return_value = mock_process_instance

        kill_process("target_process")
        assert mock_kill_children.call_count == 2


class TestCloseFileFp(unittest.TestCase):
    def test_close_file_fp_with_none(self):
        """测试传入None的情况"""
        close_file_fp(None)  # 不应该抛出异常

    def test_close_file_fp_with_file_object(self):
        """测试传入文件对象的情况"""
        mock_file = MagicMock()
        mock_file.close.return_value = None
        
        close_file_fp(mock_file)
        
        mock_file.close.assert_called_once()

    def test_close_file_fp_with_file_descriptor(self):
        """测试传入文件描述符的情况"""
        with patch('os.close') as mock_os_close:
            test_fd = 123
            close_file_fp(test_fd)
            mock_os_close.assert_called_once_with(test_fd)

    def test_close_file_fp_with_file_object_close_fails(self):
        """测试文件对象close失败的情况"""
        mock_file = MagicMock()
        mock_file.close.side_effect = AttributeError("close failed")
        
        # 不应该抛出异常
        close_file_fp(mock_file)
        mock_file.close.assert_called_once()

    def test_close_file_fp_with_fd_close_fails(self):
        """测试文件描述符close失败的情况"""
        with patch('os.close') as mock_os_close:
            mock_os_close.side_effect = OSError("close failed")
            test_fd = 123
            
            # 不应该抛出异常
            close_file_fp(test_fd)
            mock_os_close.assert_called_once_with(test_fd)

    @patch('builtins.open', unittest.mock.mock_open())
    def test_close_file_fp_with_real_file_object(self):
        """测试真实文件对象"""
        with open('test.txt', 'w') as real_file:
            real_file.close = MagicMock(wraps=real_file.close)
            close_file_fp(real_file)
            real_file.close.assert_called_once()


class TestClearingResidualProcess(unittest.TestCase):
    @patch('msserviceprofiler.modelevalstate.optimizer.utils.kill_process')
    @patch('msserviceprofiler.modelevalstate.config.config.MindieConfig')
    def test_clearing_residual_process_called(self, mock_config, mock_kill):
        """最简单测试：只验证函数被调用"""
        # 设置mock返回值
        mock_config.return_value.process_name = "test_process"
        
        # 调用被测函数
        clearing_residual_process()
        
        # 只验证kill_process被调用即可
        mock_kill.assert_called_once()


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
    def test_check_success(self, mock_exists, mock_open_func):
        mock_file = MagicMock()
        mock_open_func.return_value.__enter__.return_value = mock_file
        mock_file.tell.side_effect = [0, 100]
        mock_file.read.return_value = "test output"
        
        self.benchmark.process = MagicMock()
        self.benchmark.process.poll.return_value = 0
        
        result = self.benchmark.check_success(print_log=True)
        self.assertTrue(result)

    @patch('subprocess.Popen')
    def test_run_basic(self, mock_popen):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        params = ()
        self.benchmark.run(params)
        
        mock_popen.assert_called_once()
        self.assertEqual(self.benchmark.process, mock_process)
        self.assertIsNotNone(self.benchmark.run_log)
        self.assertEqual(self.benchmark.run_log_offset, 0)

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_stop_with_del_log(self, mock_file, mock_exists):
        self.benchmark.process = MagicMock()
        self.benchmark.process.poll.return_value = None
        self.benchmark.run_log = str(self.temp_dir / "test_log.txt")
        self.benchmark.run_log_fp = 123
        
        with patch('shutil.rmtree') as mock_rmtree:
            self.benchmark.stop(del_log=True)
            
            self.benchmark.process.kill.assert_called_once()
            mock_rmtree.assert_not_called()  # Only checks that backup wasn't called with del_log=True

    def test_get_performance_index_with_common_file(self):
        output_path = Path(self.benchmark_config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create common CSV file
        common_data = {
            "OutputGenerateSpeed": ["100 tokens/s"],
            "Returned": ["99.9%"],
        }
        pd.DataFrame(common_data).to_csv(output_path / "result_common.csv", index=False)
        
        # Create perf CSV file
        perf_data = {
            "FirstTokenTime": ["50 ms"],
            "GeneratedTokenSpeed": ["200 tokens/s"],
            "DecodeTime": ["5 ms"],  # Fixed typo: DecodeTime instead of DecodeTime
        }
        pd.DataFrame(perf_data).to_csv(output_path / "result_perf.csv", index=False)
        
        result = self.benchmark.get_performance_index()
        
        self.assertEqual(result.generate_speed, 100)  # Uses common_generate_speed
        self.assertAlmostEqual(result.time_to_first_token, 0.05)
        self.assertAlmostEqual(result.time_per_output_token, 0.005)
        self.assertAlmostEqual(result.success_rate, 0.999)

    def test_get_performance_index_with_only_common_file(self):
        output_path = Path(self.benchmark_config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create only common CSV file
        common_data = {
            "OutputGenerateSpeed": ["100 tokens/s"],
            "Returned": ["99.9%"],
        }
        pd.DataFrame(common_data).to_csv(output_path / "result_common.csv", index=False)
        
        with self.assertRaises(ValueError) as context:
            self.benchmark.get_performance_index()
        self.assertIn("Not Found first_token_time", str(context.exception))

    def test_get_performance_index_with_perf_file(self):
        output_path = Path(self.benchmark_config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create test CSV file
        test_data = {
            "FirstTokenTime": ["50 ms"],
            "GeneratedTokenSpeed": ["200 tokens/s"],
            "DecodeTime": ["5 ms"],
        }
        df = pd.DataFrame(test_data)
        df.to_csv(output_path / "result_perf.csv", index=False)
        
        self.benchmark.throughput_type = "perf"
        result = self.benchmark.get_performance_index()
        
        self.assertEqual(result.generate_speed, 200)
        self.assertEqual(result.time_to_first_token, 0.05)
        self.assertEqual(result.time_per_output_token, 0.005)

    @patch('os.path.exists', return_value=True)
    def test_check_success_process_not_finished(self, mock_exists):
        self.benchmark.process = MagicMock()
        self.benchmark.process.poll.return_value = None
        
        result = self.benchmark.check_success()
        self.assertFalse(result)

    @patch('os.path.exists', return_value=True)
    def test_check_success_process_failed(self, mock_exists):
        self.benchmark.process = MagicMock()
        self.benchmark.process.poll.return_value = 1
        
        with self.assertRaises(subprocess.SubprocessError):
            self.benchmark.check_success()


def test_stop_process_killed_successfully():
    simulator = VllmSimulator(settings.simulator)
    simulator.process = MagicMock()
    simulator.process.poll.return_value = None
    simulator.process.kill.return_value = None
    simulator.process.wait.return_value = None
    simulator.process.send_signal.return_value = None
    simulator.stop()
    # No exception should be raised


def test_stop_process_kill_timeout():
    simulator = VllmSimulator(settings.simulator)
    simulator.process = MagicMock()
    simulator.process.poll.return_value = None
    simulator.process.kill.return_value = None
    simulator.process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=10)
    simulator.process.send_signal.return_value = None
    simulator.stop()


def test_stop_process_kill_failed():
    simulator = VllmSimulator(settings.simulator)
    simulator.process = MagicMock()
    simulator.process.poll.return_value = None
    simulator.process.kill.return_value = None
    simulator.process.wait.return_value = None
    simulator.process.send_signal.return_value = None
    simulator.stop()


class TestVllmSimulator:

    @pytest.fixture
    def pre_simulator(self):
        simulator = VllmSimulator(settings.simulator)
        simulator.mindie_log = "mock_log_file.log"
        simulator.mindie_log_offset = 0
        simulator.process = MagicMock()
        return simulator

    @patch("builtins.open", new_callable=MagicMock)
    def test_check_success_with_success_message(self, mock_open_local, pre_simulator):
        # Mock the file read to return a success message
        mock_open_local.return_value.__enter__.return_value.read.return_value = "Application startup complete."
        assert pre_simulator.check_success() is True

    @patch("builtins.open", new_callable=MagicMock)
    def test_check_success_with_process_finished(self, mock_open_local, pre_simulator):
        # Mock the file read to return no success message and the process to be finished
        mock_open_local.return_value.__enter__.return_value.read.return_value = "Some other message"
        pre_simulator.process.poll.return_value = 0
        pre_simulator.process.returncode = 1
        with pytest.raises(subprocess.SubprocessError):
            pre_simulator.check_success()

    @patch("builtins.open", new_callable=MagicMock)
    def test_check_success_with_process_not_finished(self, mock_open_local, pre_simulator):
        # Mock the file read to return no success message and the process to not be finished
        mock_open_local.return_value.__enter__.return_value.read.return_value = "Some other message"
        pre_simulator.process.poll.return_value = None
        assert pre_simulator.check_success() is False



class TestVllmSimulatorRun:
    @pytest.fixture
    def pre_simulator(self):
        return VllmSimulator(settings.simulator)

    @patch.object(VllmSimulator, 'check_env')
    @patch.object(VllmSimulator, 'start_server')
    def test_run_success(self, mock_start_server, mock_check_env, pre_simulator):
        # Arrange
        run_params = (OptimizerConfigField(),)
        mock_check_env.return_value = None
        mock_start_server.return_value = None

        # Act
        pre_simulator.run(run_params)

        # Assert
        mock_check_env.assert_called_once()
        mock_start_server.assert_called_once_with(run_params)


class TestScheduleWithMultiMachine:
    @staticmethod
    def test_back_up_with_bak_path(schedule_with_multi_machine):
        # 测试当bak_path存在时的情况
        schedule_with_multi_machine.back_up()
        # 验证bak_path是否被正确设置
        assert schedule_with_multi_machine.simulator.bak_path == schedule_with_multi_machine.bak_path.joinpath("1")
        assert schedule_with_multi_machine.benchmark.bak_path == schedule_with_multi_machine.bak_path.joinpath("1")
        for rpc in schedule_with_multi_machine.rpc_clients:
            assert rpc.simulator.bak_path == schedule_with_multi_machine.bak_path.joinpath("1")

    @pytest.fixture
    def schedule_with_multi_machine(self, tmpdir):
        # 创建一个ScheduleWithMultiMachine的实例
        schedule = ScheduleWithMultiMachine(MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                            bak_path=Path(tmpdir))
        # 模拟需要的属性和方法
        schedule.simulator = MagicMock()
        schedule.benchmark = MagicMock()
        schedule.rpc_clients = [MagicMock(), MagicMock()]
        for rpc in schedule.rpc_clients:
            rpc.simulator = MagicMock()
        return schedule


class TestScheduleWithMultiMachineMonitoringStatus:

    @pytest.fixture
    def schedule_with_multi_machine(self, tmpdir):
        schedule = ScheduleWithMultiMachine(MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                            bak_path=Path(tmpdir))
        schedule.simulator = MagicMock()
        schedule.simulator.process = MagicMock()
        schedule.rpc_clients = [MagicMock() for _ in range(3)]
        schedule.benchmark = MagicMock()
        schedule.stop_target_server = MagicMock()
        return schedule

    @patch('time.sleep', return_value=None)
    def test_monitoring_status_all_poll_none(self, mock_sleep, schedule_with_multi_machine):
        schedule_with_multi_machine.simulator.process.poll.return_value = None
        for rpc in schedule_with_multi_machine.rpc_clients:
            rpc.process_poll.return_value = None
        schedule_with_multi_machine.benchmark.check_success.return_value = True

        schedule_with_multi_machine.monitoring_status()

        assert mock_sleep.call_count == 0

    @patch('time.sleep', return_value=None)
    def test_monitoring_status_some_poll_not_none(self, mock_sleep, schedule_with_multi_machine):
        schedule_with_multi_machine.simulator.process.poll.return_value = None
        schedule_with_multi_machine.rpc_clients[0].process_poll.return_value = 0
        schedule_with_multi_machine.rpc_clients[1].process_poll.return_value = None
        schedule_with_multi_machine.rpc_clients[2].process_poll.return_value = 1
        schedule_with_multi_machine.benchmark.check_success.return_value = False

        with pytest.raises(subprocess.SubprocessError):
            schedule_with_multi_machine.monitoring_status()

        assert mock_sleep.call_count == 0


def test_run_simulate(tmpdir):
    # 创建一个ScheduleWithMultiMachine实例
    schedule = ScheduleWithMultiMachine(MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                        bak_path=Path(tmpdir))
    schedule.simulator = MagicMock()
    schedule.simulator.process = MagicMock()
    schedule.benchmark = MagicMock()
    schedule.benchmark.prepare.return_value = True
    schedule.stop_target_server = MagicMock()
    schedule.run = MagicMock()
    schedule.rpc_clients[0].process_poll.return_value = 0
    schedule.rpc_clients[0].check_success.return_value = True
    schedule.rpc_clients[0].run_simulator.return_value = True
    schedule.wait_simulate = MagicMock()
    schedule.simulate_run_info = []
    # 创建模拟参数
    params = np.array([1.3] * len(default_support_field))

    # 模拟map_param_with_value方法的返回值
    # 调用run_simulate方法
    schedule.run_simulate(params, default_support_field)
    schedule.rpc_clients[0].check_success.assert_called_once()


@patch("msserviceprofiler.modelevalstate.optimizer.optimizer.PSOOptimizer")
@patch("msserviceprofiler.modelevalstate.optimizer.simulator.Simulator")
def test_main(simulator, psooptimizer):
    args = MagicMock()
    args.benchmark_policy = BenchMarkPolicy.benchmark.value
    args.deploy_policy = DeployPolicy.single.value
    args.backup = False
    args.load_breakpoint = False

    # 调用被测试的方法
    main(args)
    simulator.assert_called_once()
    psooptimizer.assert_called_once() 
