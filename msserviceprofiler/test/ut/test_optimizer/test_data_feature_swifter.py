import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np
from msserviceprofiler.modelevalstate.data_feature.dataset import MyDataSet
from msserviceprofiler.modelevalstate.inference.constant import OpAlgorithm
from msserviceprofiler.modelevalstate.inference.data_format_v1 import (
    MODEL_OP_FIELD,
    MODEL_STRUCT_FIELD,
    MODEL_CONFIG_FIELD,
    MINDIE_FIELD,
    ENV_FIELD,
    HARDWARE_FIELD,
)
from msserviceprofiler.modelevalstate.inference.dataset import TOTAL_OUTPUT_LENGTH, \
    TOTAL_SEQ_LENGTH, TOTAL_PREFILL_TOKEN
from msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter import MyDataSetWithSwifter  

class TestMyDataSetWithSwifter(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.sample_data = pd.DataFrame({
            "batch_field": ["[[1,2],[3,4]]"]*3,
            "request_field": ["[[5,6],[7,8]]"]*3,
            "op_field": ["[[9,10]]"]*3,
            "model_execute_time": [0.1, 0.2, 0.3]
        })
        self.dataset = MyDataSetWithSwifter()

    @patch('msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter.logger.info')
    @patch('msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter.MyDataSetWithSwifter.proprocess_with_swifter')
    def test_preprocess_dispatch_success(self, mock_process, mock_logger):
        """测试swifter预处理成功路径"""
        expected_features = pd.DataFrame({'feat': [1,2,3]})
        expected_labels = pd.DataFrame({'label': [0.1,0.2,0.3]})
        mock_process.return_value = (expected_features, expected_labels)
        
        # 调用测试方法
        result = self.dataset.preprocess_dispatch(self.sample_data)
        
        # 验证行为
        mock_logger.assert_called_with(f"start construct_data with swifter, shape {self.sample_data.shape}")
        mock_process.assert_called_once_with(self.sample_data)
        self.assertEqual(result, (expected_features, expected_labels))

    @patch('msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter.logger.error')
    @patch('msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter.MyDataSetWithSwifter.proprocess_with_swifter')
    @patch('msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter.MyDataSet.preprocess_dispatch')
    def test_preprocess_dispatch_fallback(self, mock_parent, mock_process, mock_logger):
        """测试swifter失败时回退到父类实现"""
        # 模拟swifter处理失败
        mock_process.side_effect = Exception("Mocked swifter error")
        expected_fallback = (pd.DataFrame(), pd.DataFrame())
        mock_parent.return_value = expected_fallback
        
        # 调用测试方法
        result = self.dataset.preprocess_dispatch(self.sample_data)
        
        # 验证行为
        mock_process.assert_called_once_with(self.sample_data)
        mock_logger.assert_called_with("Failed in construct data with swifter. error: Mocked swifter error")
        mock_parent.assert_called_once_with(self.sample_data)
        self.assertEqual(result, expected_fallback)

    @patch('pandas.DataFrame.swifter.apply')
    def test_proprocess_with_swifter_empty_data(self, mock_apply):
        """测试空数据输入处理"""
        mock_apply.return_value.values = []
        with self.assertRaises(IndexError):
            self.dataset.proprocess_with_swifter(pd.DataFrame())

    @patch('msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter.logger.info')
    @patch('msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter.MyDataSetWithSwifter.proprocess_with_swifter')
    def test_preprocess_dispatch_none_input(self, mock_process, mock_logger):
        """测试None输入直接回退父类"""
        mock_parent_return = MagicMock()
        with patch.object(MyDataSetWithSwifter, 'preprocess_dispatch', 
                         side_effect=[mock_parent_return]) as mock_parent:
            result = self.dataset.preprocess_dispatch(None)
            mock_parent.assert_called_once_with(None)
            self.assertEqual(result, mock_parent_return)
            mock_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()