import unittest
from unittest.mock import patch, MagicMock, Mock
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from xmlrpc.client import ServerProxy
import tempfile
import shutil
import os

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