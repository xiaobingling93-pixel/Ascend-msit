import unittest
from unittest.mock import patch, MagicMock
import os
import json
import numpy as np
from msserviceprofiler.modelevalstate.optimizer.optimizer import Optimizer

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.base_path = '/tmp/test_optimizer'
        self.model_path = '/path/to/model'
        self.config_path = os.path.join(self.base_path, 'config.json')
        self.output_path = os.path.join(self.base_path, 'output')
        
        # 创建基本配置
        self.config = {
            'model_type': 'gpt',
            'batch_size': {'min': 1, 'max': 8},
            'tp': {'min': 1, 'max': 2},
            'pp': {'min': 1, 'max': 2},
            'request_queue_size': {'min': 1, 'max': 4},
            'max_inference_batch_size': {'min': 1, 'max': 8},
            'max_prefill_token_num': {'min': 1, 'max': 4096},
            'max_total_token_num': {'min': 1, 'max': 8192}
        }
        
    @patch('json.load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_init_with_valid_config(self, mock_open, mock_json_load):
        """测试使用有效配置初始化Optimizer"""
        mock_json_load.return_value = self.config
        
        optimizer = Optimizer(
            model_path=self.model_path,
            config_path=self.config_path,
            output_path=self.output_path
        )
        
        self.assertEqual(optimizer.model_path, self.model_path)
        self.assertEqual(optimizer.config_path, self.config_path)
        self.assertEqual(optimizer.output_path, self.output_path)

    @patch('json.load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_init_with_invalid_config(self, mock_open, mock_json_load):
        """测试使用无效配置初始化Optimizer"""
        invalid_config = self.config.copy()
        invalid_config['batch_size']['min'] = 10
        invalid_config['batch_size']['max'] = 5  # min > max，无效
        
        mock_json_load.return_value = invalid_config
        
        with self.assertRaises(ValueError):
            Optimizer(
                model_path=self.model_path,
                config_path=self.config_path,
                output_path=self.output_path
            )

    @patch('json.load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_optimize_basic_flow(self, mock_open, mock_json_load):
        """测试基本优化流程"""
        mock_json_load.return_value = self.config
        
        optimizer = Optimizer(
            model_path=self.model_path,
            config_path=self.config_path,
            output_path=self.output_path
        )
        
        # Mock evaluate方法返回一个固定的评估结果
        optimizer.evaluate = MagicMock(return_value=(100.0, 0.5, 0.3, 0.95))
        
        result = optimizer.optimize(max_evals=2)
        
        self.assertIsInstance(result, dict)
        self.assertTrue('best_config' in result)
        self.assertTrue('best_throughput' in result)

    @patch('json.load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_init_eval_state(self, mock_open, mock_json_load):
        """测试评估状态初始化"""
        mock_json_load.return_value = self.config
        
        optimizer = Optimizer(
            model_path=self.model_path,
            config_path=self.config_path,
            output_path=self.output_path
        )
        
        test_config = {
            'batch_size': 4,
            'tp': 1,
            'pp': 1,
            'request_queue_size': 2,
            'max_inference_batch_size': 4,
            'max_prefill_token_num': 2048,
            'max_total_token_num': 4096
        }
        
        # Mock subprocess和其他依赖
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            optimizer.init_eval_state(test_config)

    @patch('json.load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_evaluate_success(self, mock_open, mock_json_load):
        """测试成功的评估场景"""
        mock_json_load.return_value = self.config
        
        optimizer = Optimizer(
            model_path=self.model_path,
            config_path=self.config_path,
            output_path=self.output_path
        )
        
        test_config = {
            'batch_size': 4,
            'tp': 1,
            'pp': 1,
            'request_queue_size': 2,
            'max_inference_batch_size': 4,
            'max_prefill_token_num': 2048,
            'max_total_token_num': 4096
        }
        
        # Mock必要的依赖
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            optimizer.init_eval_state = MagicMock()
            optimizer.analyze_profiler = MagicMock(return_value=(100.0, 0.5, 0.3, 0.95))
            
            result = optimizer.evaluate(test_config)
            
            self.assertEqual(len(result), 4)
            self.assertIsInstance(result[0], (int, float))  # throughput
            self.assertIsInstance(result[1], (int, float))  # avg_prefill_latency
            self.assertIsInstance(result[2], (int, float))  # avg_decode_latency
            self.assertIsInstance(result[3], (int, float))  # success_rate

    @patch('json.load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_evaluate_failure(self, mock_open, mock_json_load):
        """测试评估失败的场景"""
        mock_json_load.return_value = self.config
        
        optimizer = Optimizer(
            model_path=self.model_path,
            config_path=self.config_path,
            output_path=self.output_path
        )
        
        test_config = {
            'batch_size': 4,
            'tp': 1,
            'pp': 1,
            'request_queue_size': 2,
            'max_inference_batch_size': 4,
            'max_prefill_token_num': 2048,
            'max_total_token_num': 4096
        }
        
        # Mock失败场景
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)  # 设置返回错误码
            optimizer.init_eval_state = MagicMock()
            
            result = optimizer.evaluate(test_config)
            
            # 验证失败时返回默认值
            self.assertEqual(result, (0, float('inf'), float('inf'), 0))

    @patch('json.load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_optimize_with_constraints(self, mock_open, mock_json_load):
        """测试带约束条件的优化"""
        mock_json_load.return_value = self.config
        
        optimizer = Optimizer(
            model_path=self.model_path,
            config_path=self.config_path,
            output_path=self.output_path
        )
        
        # Mock evaluate返回不同的结果以测试优化逻辑
        def mock_evaluate(config):
            # 根据配置返回不同的评估结果
            throughput = config['batch_size'] * 10
            latency = 1000 / throughput
            return throughput, latency, latency, 0.95
            
        optimizer.evaluate = MagicMock(side_effect=mock_evaluate)
        
        result = optimizer.optimize(
            max_evals=5,
            constraints={
                'avg_prefill_latency': 100,
                'avg_decode_latency': 100
            }
        )
        
        self.assertIsInstance(result, dict)
        self.assertTrue('best_config' in result)
        self.assertTrue('best_throughput' in result)
        
    def test_save_and_load_results(self):
        """测试结果保存和加载"""
        with patch('json.load') as mock_json_load, \
             patch('builtins.open', new_callable=unittest.mock.mock_open) as mock_open:
            
            mock_json_load.return_value = self.config
            
            optimizer = Optimizer(
                model_path=self.model_path,
                config_path=self.config_path,
                output_path=self.output_path
            )
            
            # 创建测试结果
            test_results = {
                'best_config': {
                    'batch_size': 4,
                    'tp': 1,
                    'pp': 1
                },
                'best_throughput': 100.0
            }
            
            # Mock json.dump以验证保存操作
            with patch('json.dump') as mock_json_dump:
                optimizer.save_results(test_results)
                mock_json_dump.assert_called_once()
                
            # Mock json.load以验证加载操作
            mock_json_load.return_value = test_results
            loaded_results = optimizer.load_results()
            
            self.assertEqual(loaded_results, test_results)

if __name__ == '__main__':
    unittest.main()
