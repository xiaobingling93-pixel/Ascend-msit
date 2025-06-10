import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
from msserviceprofiler.modelevalstate.optimizer.analyze_profiler import analyze

class TestAnalyzeFunction(unittest.TestCase):
    def setUp(self):
        """准备测试数据"""
        # 基础测试数据
        self.request_data = pd.DataFrame({
            'http_rid': [0, 1, 2],
            'reply_token_size': [10, np.nan, 20],
            'first_token_latency': [100, 200, 300],
            'execution_time(microsecond)': [1000, 2000, 3000],
            'start_time_httpReq(microsecond)': [500, 1500, 2500]
        })
        
        self.batch_data = pd.DataFrame({
            'name': ['modelExec', 'modelExec', 'modelExec'],
            'batch_type': ['prefill', 'decode', 'prefill'],
            'reqinfo': ['0,100,2,200', '1,300', '2,400'],
            'start_time(microsecond)': [1000, 2000, 3000]
        })
        
        self.simulate_data = pd.DataFrame({
            'simulate_time': [1000000, 2000000, 3000000]
        })

    @patch('pandas.read_csv')
    @patch('msserviceprofiler.modelevalstate.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_normal_flow(self, mock_find_csv, mock_read_csv):
        """测试正常流程"""
        # 设置mock返回值
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            self.request_data,
            self.batch_data,
            self.simulate_data
        ]
        
        # 执行分析
        throughput, avg_prefill, avg_decode, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        # 验证结果
        self.assertIsInstance(throughput, float)
        self.assertIsInstance(avg_prefill, float)
        self.assertIsInstance(avg_decode, float)
        self.assertIsInstance(success_rate, float)
        self.assertGreater(throughput, 0)
        self.assertGreater(avg_prefill, 0)
        self.assertGreater(avg_decode, 0)
        self.assertLessEqual(success_rate, 1.0)
        self.assertGreaterEqual(success_rate, 0.0)

    @patch('pandas.read_csv')
    @patch('msserviceprofiler.modelevalstate.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_empty_data(self, mock_find_csv, mock_read_csv):
        """测试空数据集"""
        empty_request = pd.DataFrame(columns=self.request_data.columns)
        empty_batch = pd.DataFrame(columns=self.batch_data.columns)
        empty_simulate = pd.DataFrame(columns=['simulate_time'])
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            empty_request,
            empty_batch,
            empty_simulate
        ]
        
        throughput, avg_prefill, avg_decode, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        # 验证空数据的处理结果
        self.assertEqual(success_rate, 0.0)
        self.assertEqual(throughput, 0.0)
        self.assertEqual(avg_prefill, 0.0)
        self.assertEqual(avg_decode, 0.0)

    @patch('pandas.read_csv')
    @patch('msserviceprofiler.modelevalstate.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_all_successful_requests(self, mock_find_csv, mock_read_csv):
        """测试全部成功请求的场景"""
        # 创建全部成功的请求数据
        all_success_request = self.request_data.copy()
        all_success_request['reply_token_size'] = 10  # 设置所有请求都有回复
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            all_success_request,
            self.batch_data,
            self.simulate_data
        ]
        
        _, _, _, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        self.assertEqual(success_rate, 1.0)

    @patch('pandas.read_csv')
    @patch('msserviceprofiler.modelevalstate.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_mismatched_rows(self, mock_find_csv, mock_read_csv):
        """测试行数不匹配的情况"""
        # 创建行数不匹配的模拟数据
        mismatched_simulate = pd.DataFrame({
            'simulate_time': [1000000, 2000000]  # 只有2行
        })
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            self.request_data,
            self.batch_data,
            mismatched_simulate
        ]
        
        with self.assertRaises(ValueError):
            analyze(input_path_1='/fake/path1', input_path_2='/fake/path2')

    @patch('pandas.read_csv')
    @patch('msserviceprofiler.modelevalstate.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_invalid_data_format(self, mock_find_csv, mock_read_csv):
        """测试无效的数据格式"""
        # 创建缺少必要列的数据
        invalid_request = pd.DataFrame({
            'http_rid': [0, 1, 2]  # 缺少其他必要列
        })
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            invalid_request,
            self.batch_data,
            self.simulate_data
        ]
        
        with self.assertRaises(KeyError):
            analyze(input_path_1='/fake/path1', input_path_2='/fake/path2')

    @patch('pandas.read_csv')
    @patch('msserviceprofiler.modelevalstate.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_edge_case_single_request(self, mock_find_csv, mock_read_csv):
        """测试单个请求的边界情况"""
        # 创建只有一个请求的数据
        single_request = pd.DataFrame({
            'http_rid': [0],
            'reply_token_size': [10],
            'first_token_latency': [100],
            'execution_time(microsecond)': [1000],
            'start_time_httpReq(microsecond)': [500]
        })
        
        single_batch = pd.DataFrame({
            'name': ['modelExec'],
            'batch_type': ['prefill'],
            'reqinfo': ['0,100'],
            'start_time(microsecond)': [1000]
        })
        
        single_simulate = pd.DataFrame({
            'simulate_time': [1000000]
        })
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            single_request,
            single_batch,
            single_simulate
        ]
        
        throughput, avg_prefill, avg_decode, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        self.assertEqual(success_rate, 1.0)
        self.assertGreater(throughput, 0)

    @patch('pandas.read_csv')
    @patch('msserviceprofiler.modelevalstate.optimizer.analyze_profiler.find_first_simulate_csv')
    def test_no_successful_requests(self, mock_find_csv, mock_read_csv):
        """测试没有成功请求的场景"""
        # 创建全部失败的请求数据
        all_failed_request = self.request_data.copy()
        all_failed_request['reply_token_size'] = np.nan  # 设置所有请求都没有回复
        
        mock_find_csv.return_value = 'simulated.csv'
        mock_read_csv.side_effect = [
            all_failed_request,
            self.batch_data,
            self.simulate_data
        ]
        
        throughput, avg_prefill, avg_decode, success_rate = analyze(
            input_path_1='/fake/path1',
            input_path_2='/fake/path2'
        )
        
        self.assertEqual(success_rate, 0.0)
        self.assertEqual(throughput, 0.0)

    def test_file_not_found(self):
        """测试文件不存在的情况"""
        with self.assertRaises(FileNotFoundError):
            analyze(
                input_path_1='/nonexistent/path1',
                input_path_2='/nonexistent/path2'
            )

if __name__ == '__main__':
    unittest.main()
