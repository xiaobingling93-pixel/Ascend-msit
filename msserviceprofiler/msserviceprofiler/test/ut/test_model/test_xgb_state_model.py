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
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import xgboost
import tempfile
import shutil
import os


# 测试类
class TestStateXgbModel(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.save_model_path = Path(self.temp_dir) / "model.bin"
        self.middle_save_path = Path(self.temp_dir) / "plots"
        self.middle_save_path.mkdir(exist_ok=True)

        # 创建模拟数据集
        self.mock_dataset = MagicMock()
        self.mock_dataset.train_x = np.array([[1, 2], [3, 4]])
        self.mock_dataset.train_y = np.array([1, 2])
        self.mock_dataset.test_x = np.array([[5, 6], [7, 8]])
        self.mock_dataset.test_y = np.array([3, 4])
        self.mock_dataset.test_y.values.flatten.return_value = np.array([3, 4])

        # 创建模拟模型
        self.mock_model = MagicMock(spec=xgboost.Booster)
        self.mock_model.predict.return_value = np.array([3.5, 4.5])
        self.mock_model.feature_names = ['f1', 'f2']
        self.mock_model.get_score.return_value = {'f1': 0.8, 'f2': 0.2}

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_train_fit(self, mock_dmatrix, mock_train):
        # 设置mock
        mock_train.return_value = self.mock_model
        mock_dmatrix.return_value = MagicMock()

        # 初始化模型
        model = StateXgbModel(
            save_model_path=self.save_model_path,
            show_test_data_prediction=False,
            show_feature_importance=False
        )

        # 执行训练
        rmse = model.train(self.mock_dataset, train_type='fit')

        # 验证调用
        mock_train.assert_called_once()
        self.mock_model.save_model.assert_called_once_with(self.save_model_path)
        self.assertIsInstance(rmse, float)

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_train_partial_fit(self, mock_dmatrix, mock_train):
        # 设置mock
        mock_train.return_value = self.mock_model
        mock_dmatrix.return_value = MagicMock()

        # 初始化模型
        model = StateXgbModel(
            load_model_path=self.save_model_path,
            save_model_path=self.save_model_path,
            show_test_data_prediction=False,
            show_feature_importance=False
        )

        # 执行训练
        rmse = model.train(self.mock_dataset, train_type='partial_fit')

        # 验证调用
        mock_train.assert_called_once()
        self.assertEqual(mock_train.call_args[0][0], model.update_param)
        self.mock_model.save_model.assert_called_once_with(self.save_model_path)

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    @patch('msserviceprofiler.modelevalstate.model_xgb.plot_feature_importance')
    @patch('msserviceprofiler.modelevalstate.model_xgb.plot_pred_and_test')
    def test_train_with_plots(self, mock_plot_pred, mock_plot_feat, mock_dmatrix, mock_train):
        # 设置mock
        mock_train.return_value = self.mock_model
        mock_dmatrix.return_value = MagicMock()

        # 初始化模型
        model = StateXgbModel(
            save_model_path=self.save_model_path,
            show_test_data_prediction=True,
            show_feature_importance=True
        )

        # 执行训练
        model.train(self.mock_dataset, middle_save_path=self.middle_save_path)

        # 验证绘图函数被调用
        mock_plot_pred.assert_called_once()
        mock_plot_feat.assert_called_once_with(self.mock_model, self.middle_save_path)

    def test_predict(self):
        # 初始化模型
        model = StateXgbModel(load_model_path=self.save_model_path)

        # 创建模型文件
        self.mock_model.save_model(self.save_model_path)

        # 执行预测
        data = np.array([[1, 2], [3, 4]])
        predictions = model.predict(data)

        # 验证结果
        self.assertEqual(predictions.shape, (2,))
        self.assertIsInstance(predictions, np.ndarray)

    def test_init_value_error(self):
        # 验证当save_model为True但未提供save_model_path时抛出异常
        with self.assertRaises(ValueError):
            StateXgbModel(save_model=True)

    @patch('xgboost.Booster.load_model')
    def test_predict_with_invalid_model(self, mock_load):
        # 设置mock抛出异常
        mock_load.side_effect = Exception("Model not found")

        # 初始化模型
        model = StateXgbModel(load_model_path=self.save_model_path)

        # 验证预测抛出异常
        with self.assertRaises(Exception):
            model.predict(np.array([[1, 2]]))

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_train_with_custom_params(self, mock_dmatrix, mock_train):
        # 设置mock
        mock_train.return_value = self.mock_model
        mock_dmatrix.return_value = MagicMock()

        # 自定义参数
        train_params = {'objective': 'reg:squarederror', 'max_depth': 5}
        update_params = {'process_type': 'update', 'updater': 'refresh'}

        # 初始化模型
        model = StateXgbModel(
            train_param=train_params,
            update_param=update_params,
            save_model_path=self.save_model_path
        )

        # 执行训练
        model.train(self.mock_dataset)

        # 验证参数传递正确
        mock_train.assert_called_once_with(train_params, mock_dmatrix(), evals=[(mock_dmatrix(), 'validation')])

    @patch('xgboost.plot_importance')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('builtins.open')
    def test_plot_feature_importance(self, mock_open, mock_close, mock_savefig, mock_subplots, mock_plot_importance):
        # 调用绘图函数
        plot_feature_importance(self.mock_model, self.middle_save_path)

        # 验证函数调用
        self.assertEqual(mock_savefig.call_count, 3)
        self.assertEqual(mock_close.call_count, 3)
        mock_open.assert_called_once_with(self.middle_save_path / "feature_importance.txt", 'w')

    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_pred_and_test(self, mock_close, mock_savefig, mock_figure, mock_scatter):
        # 创建模拟数据
        pred = np.array([3.5, 4.5])
        my_data = MagicMock()
        my_data.test_y.values.flatten.return_value = np.array([3, 4])

        # 调用绘图函数
        plot_pred_and_test(pred, my_data, self.middle_save_path)

        # 验证函数调用
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        self.assertEqual(mock_scatter.call_count, 2)

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_train_without_saving(self, mock_dmatrix, mock_train):
        # 设置mock
        mock_train.return_value = self.mock_model
        mock_dmatrix.return_value = MagicMock()

        # 初始化模型（不保存）
        model = StateXgbModel(
            save_model=False,
            show_test_data_prediction=False,
            show_feature_importance=False
        )

        # 执行训练
        rmse = model.train(self.mock_dataset)

        # 验证模型未保存
        self.mock_model.save_model.assert_not_called()
        self.assertIsInstance(rmse, float)


if __name__ == '__main__':
    unittest.main()