import os
import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
from msserviceprofiler.modelevalstate.optimizer.analyze_profiler import find_first_simulate_csv, analyze

class TestAnalyzeProfiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 创建测试用的数据目录
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(cls.test_dir, exist_ok=True)
        
    def setUp(self):
        # 准备测试数据
        self.request_data = pd.DataFrame({
            'http_rid': [1, 2, 3],
            'reply_token_size': [10, np.nan, 20],
            'first_token_latency': [100, 200, 300],
            'execution_time(microsecond)': [1000, 2000, 3000],
            'start_time_httpReq(microsecond)': [500, 1500, 2500]
        })
        
        self.batch_data = pd.DataFrame({
            'name': ['modelExec', 'modelExec', 'other'],
            'batch_type': ['prefill', 'decode', 'prefill'],
            'reqinfo': ['1,2,3', '4,5,6', '7,8,9'],
            'start_time(microsecond)': [1000, 2000, 3000]
        })
        
        self.simulate_data = pd.DataFrame({
            'simulate_time': [1000000, 2000000, 3000000]
        })

    def test_find_first_simulate_csv_invalid_dir(self):
        """测试无效目录路径"""
        with self.assertRaises(NotADirectoryError):
            find_first_simulate_csv('/nonexistent/directory')

    def test_find_first_simulate_csv_no_files(self):
        """测试目录中没有满足条件的文件"""
        with self.assertRaises(FileNotFoundError):
            find_first_simulate_csv(self.test_dir)

    @patch('glob.glob')
    def test_find_first_simulate_csv_multiple_files(self, mock_glob):
        """测试有多个满足条件的文件时正确返回第一个"""
        mock_files = [
            os.path.join(self.test_dir, 'simulate2.csv'),
            os.path.join(self.test_dir, 'simulate1.csv')
        ]
        mock_glob.return_value = mock_files
        result = find_first_simulate_csv(self.test_dir)
        self.assertEqual(result, os.path.join(self.test_dir, 'simulate1.csv'))

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_analyze_normal_flow(self, mock_exists, mock_read_csv):
        """测试分析函数的正常流程"""
        # Mock文件存在检查
        mock_exists.return_value = True
        
        # Mock DataFrame读取
        mock_read_csv.side_effect = [
            self.request_data,  # request.csv
            self.batch_data,    # batch.csv
            self.simulate_data  # simulate.csv
        ]
        
        try:
            throughput, avg_prefill, avg_decode, success_rate = analyze(
                '/fake/path1',
                '/fake/path2'
            )
            
            # 验证返回值是否为预期类型
            self.assertIsInstance(throughput, float)
            self.assertIsInstance(avg_prefill, float)
            self.assertIsInstance(avg_decode, float)
            self.assertIsInstance(success_rate, float)
            
            # 验证成功率计算是否正确
            self.assertAlmostEqual(success_rate, 2/3)  # 2个有效请求，总共3个请求
            
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    @patch('pandas.read_csv')
    def test_analyze_mismatched_rows(self, mock_read_csv):
        """测试CSV文件行数不匹配的情况"""
        # 创建行数不匹配的DataFrame
        mismatched_simulate_data = pd.DataFrame({
            'simulate_time': [1000000, 2000000]  # 只有2行，而batch_data有3行
        })
        
        mock_read_csv.side_effect = [
            self.request_data,
            self.batch_data,
            mismatched_simulate_data
        ]
        
        with self.assertRaises(ValueError):
            analyze('/fake/path1', '/fake/path2')

    def tearDown(self):
        # 清理测试数据
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
