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
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import os
from dataclasses import asdict

import shutil
import tempfile
import numpy as np
import joblib
import xgboost
import pandas as pd

from msserviceprofiler.modelevalstate.train.pretrain import (
    PretrainModel, TrainVersion1, pretrain
)
from msserviceprofiler.modelevalstate.train.state_param import StateParam
from msserviceprofiler.modelevalstate.data_feature.dataset import MyDataSet
from msserviceprofiler.modelevalstate.model.xgb_state_model import StateXgbModel
from msserviceprofiler.modelevalstate.data_feature.dataset import preset_category_data
from msserviceprofiler.modelevalstate.data_feature.dataset import CustomLabelEncoder


class TestPretrainModel(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.test_dir = "test_data"
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir, 0o750)
        self.input_path = Path(self.test_dir) / "input"
        self.output_path = Path(self.test_dir) / "output"
        self.feature_path = Path(self.input_path) / "test"
        self.input_path.mkdir()
        self.feature_path.mkdir()
        self.output_path.mkdir()
        # 使用测试数据
        pretrain_data_path = os.path.join(os.path.dirname(__file__), 'pretrain_data.json')
        if os.path.exists(pretrain_data_path):
            self.real_data = pd.read_csv(pretrain_data_path)
            self.real_data_file = Path(self.feature_path) / "feature.csv"
            self.real_data.to_csv(self.real_data_file, index=False)
        # 为真实测试创建数据集和模型实例
        if self.real_data is not None:
            self.real_dataset = MyDataSet()
            sp = StateParam(
                base_path=self.output_path,
                predict_field="model_execute_time",
                save_model=True,
                shuffle=True,
                plot_pred_and_real=True,
                plot_data_feature=True,
                start_num_lines=4000,
                title="MixModel without warmup with batch max seq 2 op info"
            )
            self.state_param = sp
            self.real_model = StateXgbModel(
                train_param=sp.xgb_model_train_param,
                update_param=sp.xgb_model_update_param,
                save_model_path=sp.xgb_model_save_model_path,
                load_model_path=sp.xgb_model_save_model_path,
                show_test_data_prediction=sp.xgb_model_show_test_data_prediction,
                show_feature_importance=sp.xgb_model_show_feature_importance
            )
            self.real_trainer = PretrainModel(
                state_param=sp,
                dataset=self.real_dataset,
                model=self.real_model,
                plt_data=False
            )

        # 创建临时变量
        self.dataset = MagicMock()
        self.model = MagicMock()
        self.pm = PretrainModel(
            state_param=sp,
            dataset=self.dataset,
            model=self.model
        )
        self.trainer = PretrainModel(
            state_param=sp,
            dataset=self.dataset,
            model=self.model,
            plt_data=False
        )

        self.sample_data = pd.DataFrame({
            'batch_stage': ['prefill', 'decode', 'prefill'],
            'batch_size': [1, 2, 3],
            'model_execute_time': [100, 200, 300]
        })

        self.features = pd.DataFrame({
            'batch_stage': [1, 0],
            'batch_size': [1, 2],
            'feature1': [0.5, 0.7]
        })
        self.labels = pd.Series([100, 200])

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    def test_train_with_sample_data(self):
        """测试使用样本数据训练"""
        # 配置模拟返回值
        self.dataset.construct_data.return_value = None
        self.model.train.return_value = 0.25

        # 执行训练
        rmse = self.trainer.train(lines_data=self.sample_data)

        # 验证调用
        self.dataset.construct_data.assert_called_once_with(
            self.sample_data, plt_data=False, middle_save_path=None
        )
        self.model.train.assert_called_once_with(self.dataset, middle_save_path=None)

        # 验证结果
        self.assertIn(0.25, self.trainer.rmse)

    #@unittest.skipIf(self.real_data is None, "缺少真实测试数据")
    '''def test_train_with_real_data(self):
        """使用真实数据集测试训练流程"""
        # 首先进行初始训练
        custom_encoder = CustomLabelEncoder(preset_category_data)
        dataset = MyDataSet(custom_encoder=custom_encoder)
        initial_rmse = self.real_trainer.train(lines_data=self.real_data)
        self.assertIsInstance(initial_rmse, float)
        self.assertGreater(initial_rmse, 0)

        # 验证模型文件存在
        model_file = self.state_param.xgb_model_save_model_path
        self.assertTrue(model_file.exists())

        # 验证特征编码器存在
        ohe_file = self.state_param.ohe_path
        self.assertTrue(ohe_file.exists())

        # 验证指标记录
        self.assertEqual(len(self.real_trainer.rmse), 1)

        # 保存模型状态
        initial_model = joblib.load(model_file)

        # 使用新的数据集进行增量训练
        # 创建新的训练数据（复制前50行并稍微修改）
        new_data = self.real_data.copy().head(50)
        new_data["model_execute_time"] += np.random.rand(50) * 10 - 5
        new_rmse = self.real_trainer.partial_train(lines_data=new_data)

        # 验证增量训练后指标更新
        self.assertEqual(len(self.real_trainer.rmse), 2)
        self.assertLess(new_rmse, initial_rmse * 1.5)  # 新RMSE不应该比初始值大太多

        # 验证模型已更新（文件大小/修改时间变化）
        updated_model = joblib.load(model_file)
        self.assertNotEqual(initial_model.get_params(), updated_model.get_params())'''

    #@unittest.skipIf("real_data" not in dir() or real_data is None, "缺少真实测试数据")
    '''def test_predict_with_real_data(self):
        """使用真实数据集测试预测流程"""
        # 首先进行训练
        self.real_trainer.train(lines_data=self.real_data)

        # 准备预测数据（使用50条数据）
        predict_data = self.real_data.head(50)

        # 创建预测结果保存目录
        results_dir = self.test_dir / "prediction_results"
        results_dir.mkdir()

        # 执行预测
        predictions = self.real_trainer.predict(
            lines_data=predict_data,
            save_path=results_dir
        )

        # 验证返回的预测结果
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), 50)
        self.assertIn("predict_value", predictions.columns)
        self.assertIn("real_value", predictions.columns)

        # 验证预测结果文件
        prediction_file = results_dir / "prediction_data.csv"
        self.assertTrue(prediction_file.exists())
        saved_predictions = pd.read_csv(prediction_file)
        self.assertEqual(len(saved_predictions), 50)

        # 验证可视化文件
        plot_file = results_dir / "predict_vs_real.png"
        self.assertTrue(plot_file.exists())'''

    def test_get_up_ud(self):
        class Node:
            def __init__(self, stage, batch_size, time):
                self.stage = stage
                self.batch_size = batch_size
                self.model_execute_time = time

        nodes = [
            Node('prefill', 1, 100),
            Node('decode', 2, 200)
        ]
        # 测试方法
        prefill, decode = PretrainModel.get_up_ud(
            tuple(nodes), "model_execute_time"
        )
        # 验证结果
        self.assertEqual(prefill[list(prefill)[0]], [10000.0])
        self.assertEqual(decode[list(decode)[0]], [5000.0])

    def test_model_backup(self):
        """测试模型备份功能"""
        # 创建模型文件
        model_file = self.state_param.xgb_model_save_model_path
        model_file.parent.mkdir(parents=True, exist_ok=True)
        with open(model_file, "w") as f:
            f.write("dummy model data")

        # 执行备份
        self.trainer.bak_model(increment_stage="test_stage")

        # 验证备份文件存在
        bak_file = self.state_param.bak_dir / "test_stage" / model_file.name
        self.assertTrue(bak_file.exists())

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_metric_visualization(self, mock_close, mock_savefig):
        """测试指标可视化"""
        # 设置训练指标
        self.trainer.rmse = [0.25, 0.18, 0.15]
        self.trainer.r2 = [0.92, 0.94, 0.96]
        self.trainer.mape = [0.05, 0.04, 0.03]

        # 创建保存目录
        save_path = Path(self.test_dir) / "metrics"
        save_path.mkdir()

        # 执行可视化
        self.trainer.plot_metric(save_path=save_path)

        # 验证图片保存被调用
        mock_savefig.assert_called_once_with(save_path / "metric.png")

        # 验证图片文件"存在"（通过模拟）
        self.assertTrue(mock_savefig.called)

    def test_pretrain_workflow(self):
        """测试完整的训练流程"""
        # 执行训练
        pretrain(str(self.input_path), str(self.output_path))

        # 验证输出目录结构
        self.assertTrue((self.output_path / "model").exists())
        self.assertTrue((self.output_path / "step").exists())
        self.assertTrue((self.output_path / "bak").exists())

        # 验证模型文件存在
        self.assertTrue((self.output_path / "model" / "xgb_model.ubj").exists())

        # 验证缓存数据存在
        self.assertTrue((self.output_path / "cache" / "train_data.csv").exists())

    def test_get_decode_and_prefill_time(self):
        # 准备测试数据
        class Node:
            def __init__(self, stage, batch_size, time):
                self.stage = stage
                self.batch_size = batch_size
                self.model_execute_time = time

        nodes = [
            Node('prefill', 1, 100),
            Node('decode', 2, 200),
            Node('prefill', 3, 300)
        ]
        # 测试方法
        prefill, decode = PretrainModel.get_decode_and_prefill_time(
            tuple(nodes), "model_execute_time"
        )
        # 验证结果
        self.assertEqual(prefill[list(prefill)[0]], [100])
        self.assertEqual(prefill[list(prefill)[1]], [300])
        self.assertEqual(decode[list(decode)[0]], [200])

    def test_train(self):
        with patch.object(self.dataset, 'construct_data') as mock_construct:
            with patch.object(self.model, 'train') as mock_train:
                mock_train.return_value = 0.5

                # 调用方法
                self.pm.train(self.sample_data)

                # 验证调用
                mock_construct.assert_called_once_with(
                    self.sample_data, plt_data=False, middle_save_path=None
                )
                mock_train.assert_called_once_with(
                    self.dataset, middle_save_path=None
                )
                self.assertEqual(self.pm.rmse, [0.5])

    def test_partial_train(self):
        with patch.object(self.dataset.custom_encoder, 'fit') as mock_fit:
            with patch.object(self.dataset, 'construct_data') as mock_construct:
                with patch.object(self.model, 'train') as mock_train:
                    mock_train.return_value = 0.3

                    # 调用方法
                    self.pm.partial_train(self.sample_data)

                    # 验证调用
                    mock_fit.assert_called_once_with(load=True)
                    mock_construct.assert_called_once_with(
                        self.sample_data, plt_data=False, middle_save_path=None
                    )
                    mock_train.assert_called_once_with(
                        self.dataset, train_type="partial_fit", middle_save_path=None
                    )
                    self.assertEqual(self.pm.rmse, [0.3])

    def test_initialization(self):
        """测试训练器初始化"""
        self.assertIs(self.trainer.state_param, self.state_param)
        self.assertIs(self.trainer.dataset, self.dataset)
        self.assertIs(self.trainer.model, self.model)
        self.assertFalse(self.trainer.plt_data)


if __name__ == '__main__':
    unittest.main()
