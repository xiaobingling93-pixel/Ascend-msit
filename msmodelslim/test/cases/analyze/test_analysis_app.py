# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
综合测试用例：验证分析模块的完整功能覆盖
包括CLI、APP和分析服务模块的所有功能
目标覆盖率：>80%
"""

import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from msmodelslim.app.analysis_service.layer_selector import AnalysisResult  # 替换为实际的 AnalysisResult 类
from msmodelslim.app.analysis_service.layer_selector import \
    LayerSelectorAnalysisService  # 替换为实际包含 _print_analysis_results 的类
from msmodelslim.app.base import DeviceType
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import clean_output


class TestComprehensiveAnalysisCoverage(unittest.TestCase):
    """综合测试分析模块的所有功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = Path(self.temp_dir) / "lab_calib"
        self.dataset_dir.mkdir()

        # 创建模拟的校准数据集文件
        self.calib_file = self.dataset_dir / "boolq.jsonl"
        with open(self.calib_file, 'w') as f:
            f.write('{"data": "mock calibration data"}')

        # 创建模型文件
        self.model_path = Path(self.temp_dir) / "model"
        self.model_path.mkdir()

    def tearDown(self):
        """测试后的清理工作"""
        shutil.rmtree(self.temp_dir)


class TestCLIAnalysisModule(TestComprehensiveAnalysisCoverage):
    """测试CLI分析模块"""

    @patch('msmodelslim.cli.analysis.__main__.LayerAnalysisApplication')
    @patch('msmodelslim.cli.analysis.__main__.ModelFactory')
    @patch('msmodelslim.cli.analysis.__main__.FileDatasetLoader')
    @patch('msmodelslim.cli.analysis.__main__.get_valid_read_path')
    def test_main_function_with_exception(self, mock_get_path, mock_dataset_loader,
                                          mock_model_factory, mock_app):
        """测试main函数异常处理"""
        from msmodelslim.cli.analysis.__main__ import main

        # 准备模拟对象，让analyze抛出异常
        mock_get_path.return_value = self.dataset_dir
        mock_dataset_instance = MagicMock()
        mock_dataset_loader.return_value = mock_dataset_instance
        mock_model_factory_instance = MagicMock()
        mock_model_factory.create.return_value = mock_model_factory_instance
        mock_app_instance = MagicMock()
        mock_app.return_value = mock_app_instance
        mock_app_instance.analyze.side_effect = Exception("Test exception")

        # 创建模拟参数
        args = Namespace(
            model_type="Qwen2.5-7B-Instruct",
            model_path=str(self.model_path),
            pattern=["*"],
            device="npu",
            metrics="std",
            calib_dataset="boolq.jsonl",
            topk=15,
            trust_remote_code=False
        )

        # 执行测试，期望抛出异常
        with self.assertRaises(Exception):
            main(args)

    @patch('msmodelslim.cli.analysis.__main__.os.path.dirname')
    @patch('msmodelslim.cli.analysis.__main__.os.path.abspath')
    @patch('msmodelslim.cli.analysis.__main__.os.path.join')
    @patch('msmodelslim.cli.analysis.__main__.get_valid_read_path')
    def test_get_dataset_dir(self, mock_get_path, mock_join, mock_abspath, mock_dirname):
        """测试get_dataset_dir函数"""
        from msmodelslim.cli.analysis.__main__ import get_dataset_dir

        # 准备模拟
        mock_dirname.return_value = "/path/to/cli"
        mock_abspath.return_value = "/absolute/path/to/lab_calib"
        mock_join.return_value = "/absolute/path/to/lab_calib"
        mock_get_path.return_value = Path("/absolute/path/to/lab_calib")

        # 执行测试
        result = get_dataset_dir()

        # 验证结果
        self.assertEqual(result, Path("/absolute/path/to/lab_calib"))
        mock_get_path.assert_called_once_with("/absolute/path/to/lab_calib", is_dir=True)


class TestAppAnalysisModule(TestComprehensiveAnalysisCoverage):
    """测试APP分析模块"""

    def test_layer_analysis_application_init(self):
        """测试LayerAnalysisApplication初始化"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication

        mock_service = MagicMock()
        mock_factory = MagicMock()

        app = LayerAnalysisApplication(mock_service, mock_factory)

        self.assertEqual(app.analysis_service, mock_service)
        self.assertEqual(app.model_factory, mock_factory)

    def test_analyze_with_valid_parameters(self):
        """测试analyze方法使用有效参数"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication

        # 创建模拟对象
        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()
        mock_result = MagicMock()

        # 设置模拟返回值
        mock_model_factory.return_value.return_value = mock_model_adapter
        mock_service.analyze.return_value = mock_result

        app = LayerAnalysisApplication(mock_service, mock_model_factory)

        # 调用analyze方法
        result = app.analyze(
            model_type="Qwen2.5-7B-Instruct",
            model_path=self.model_path,
            patterns=["*"],
            device=DeviceType.NPU,
            metrics="std",
            calib_dataset="boolq.jsonl",
            topk=15,
            trust_remote_code=False
        )

        # 验证结果
        self.assertEqual(result, mock_result)

        # 验证模型工厂调用
        mock_model_factory.assert_called_once()
        mock_model_factory.return_value.assert_called_once_with(
            model_type="Qwen2.5-7B-Instruct",
            ori_path=self.model_path,
            device=DeviceType.NPU,
            trust_remote_code=False
        )

        # 验证分析服务调用
        mock_service.analyze.assert_called_once_with(
            model=mock_model_adapter,
            patterns=["*"],
            analysis_config={
                'metrics': 'std',
                'calib_dataset': 'boolq.jsonl',
                'method_params': {}
            }
        )

        # 验证结果导出
        mock_service.export_results.assert_called_once_with(result, 15)

    def test_analyze_with_enum_metrics(self):
        """测试analyze方法使用枚举metrics"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics

        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()
        mock_result = MagicMock()

        mock_model_factory.return_value.return_value = mock_model_adapter
        mock_service.analyze.return_value = mock_result

        app = LayerAnalysisApplication(mock_service, mock_model_factory)

        result = app.analyze(
            model_type="Qwen2.5-7B-Instruct",
            model_path=self.model_path,
            patterns=["*"],
            device=DeviceType.NPU,
            metrics=AnalysisMetrics.KURTOSIS,
            calib_dataset="boolq.jsonl",
            topk=15,
            trust_remote_code=False
        )

        # 验证metrics被正确转换为字符串
        mock_service.analyze.assert_called_once_with(
            model=mock_model_adapter,
            patterns=["*"],
            analysis_config={
                'metrics': 'kurtosis',
                'calib_dataset': 'boolq.jsonl',
                'method_params': {}
            }
        )

    def test_analyze_parameter_validation(self):
        """测试analyze方法的参数验证"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication

        mock_service = MagicMock()
        mock_model_factory = MagicMock()

        app = LayerAnalysisApplication(mock_service, mock_model_factory)

        # 测试无效的model_type类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type=123,  # 应该是字符串
                model_path=self.model_path,
                patterns=["*"],
                device=DeviceType.NPU,
                metrics="std",
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

        # 测试无效的model_path类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path="not_a_path",  # 应该是Path对象
                patterns=["*"],
                device=DeviceType.NPU,
                metrics="std",
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

        # 测试无效的patterns类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=self.model_path,
                patterns="not_a_list",  # 应该是列表
                device=DeviceType.NPU,
                metrics="std",
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

        # 测试无效的device类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=self.model_path,
                patterns=["*"],
                device="invalid_device",  # 应该是DeviceType枚举
                metrics="std",
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

        # 测试无效的metrics值
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=self.model_path,
                patterns=["*"],
                device=DeviceType.NPU,
                metrics="invalid_metrics",  # 无效的metrics值
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

        # 测试无效的calib_dataset格式
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=self.model_path,
                patterns=["*"],
                device=DeviceType.NPU,
                metrics="std",
                calib_dataset="invalid.txt",  # 无效的文件格式
                topk=15,
                trust_remote_code=False
            )

        # 测试无效的topk值
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=self.model_path,
                patterns=["*"],
                device=DeviceType.NPU,
                metrics="std",
                calib_dataset="boolq.jsonl",
                topk=0,  # 无效的topk值
                trust_remote_code=False
            )

    def test_internal_analyze_method(self):
        """测试内部_analyze方法"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication

        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()
        mock_result = MagicMock()

        mock_model_factory.return_value.return_value = mock_model_adapter
        mock_service.analyze.return_value = mock_result

        app = LayerAnalysisApplication(mock_service, mock_model_factory)

        # 调用内部方法
        result = app._analyze(
            model_type="Qwen2.5-7B-Instruct",
            model_path=self.model_path,
            patterns=["*"],
            device=DeviceType.NPU,
            metrics="std",
            calib_dataset="boolq.jsonl",
            topk=15,
            trust_remote_code=False
        )

        self.assertEqual(result, mock_result)


class TestAnalysisServiceModule(TestComprehensiveAnalysisCoverage):
    """测试分析服务模块"""

    def test_layer_selector_analysis_service_init(self):
        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        self.assertEqual(service.dataset_loader, mock_dataset_loader)

    def test_analyze_with_invalid_model(self):
        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        with self.assertRaises(SchemaValidateError):
            service.analyze(
                model="not_a_model_adapter",  # 无效的model类型
                patterns=['*'],
                analysis_config={}
            )

    def test_analyze_with_invalid_patterns(self):
        """测试analyze方法使用无效的patterns"""

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        mock_model = MagicMock()
        mock_model.device = MagicMock()
        mock_model.device.value = 'cpu'

        with self.assertRaises(SchemaValidateError):
            service.analyze(
                model=mock_model,
                patterns="not_a_list",  # 无效的patterns类型
                analysis_config={}
            )

    def test_prepare_calibration_data_no_dataset(self):
        """测试_prepare_calibration_data方法无校准数据集"""
        from msmodelslim.app.base.model import BaseModelAdapter

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        mock_model = MagicMock()
        mock_model.__class__ = BaseModelAdapter  # 设置为BaseModelAdapter类型

        result = service._prepare_calibration_data(mock_model, None)

        self.assertIsNone(result)

    def test_get_target_layers(self):
        """测试_get_target_layers方法"""

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        mock_model = MagicMock()

        with patch('msmodelslim.app.analysis_service.layer_selector.AnalysisTargetMatcher') as mock_matcher:
            mock_matcher.get_linear_conv_layers.return_value = ['layer1', 'layer2', 'layer3']
            mock_matcher.filter_layers_by_patterns.return_value = ['layer1', 'layer2']

            result = service._get_target_layers(mock_model, ['layer*'])

        mock_matcher.get_linear_conv_layers.assert_called_once_with(mock_model)
        mock_matcher.filter_layers_by_patterns.assert_called_once_with(['layer1', 'layer2', 'layer3'], ['layer*'])
        self.assertEqual(result, ['layer1', 'layer2'])

    def test_export_results(self):
        """测试export_results方法"""

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        result = AnalysisResult(
            layer_scores=[{'name': 'layer1', 'score': 2.0}, {'name': 'layer2', 'score': 1.0}],
            method='test',
            patterns=['*']
        )

        with patch.object(service, '_print_analysis_results') as mock_print:
            service.export_results(result, 10)

        mock_print.assert_called_once_with(result, 10)

    def test_analyze_normal_execution(self):
        """测试analyze方法的正常执行流程"""
        from msmodelslim.app.base.model import BaseModelAdapter

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        # 创建模拟BaseModelAdapter对象
        mock_model = MagicMock()
        mock_model.__class__ = BaseModelAdapter  # 设置为BaseModelAdapter类型
        mock_model.device = MagicMock()
        mock_model.device.value = 'cpu'

        # 模拟各种依赖方法
        with patch.object(service, '_prepare_calibration_data', return_value=None) as mock_prep_data:
            with patch.object(service, '_get_target_layers', return_value=['layer1', 'layer2']) as mock_get_layers:
                with patch.object(service, '_run_analysis',
                                  return_value=[{'name': 'layer1', 'score': 1.0}]) as mock_run_analysis:
                    with patch('msmodelslim.app.analysis_service.layer_selector.AnalysisMethodFactory') as mock_factory:
                        mock_method = MagicMock()
                        mock_method.name = 'std'
                        mock_factory.create_method.return_value = mock_method

                        # 执行测试
                        result = service.analyze(
                            model=mock_model,
                            patterns=['*'],
                            analysis_config={
                                'metrics': 'std',
                                'calib_dataset': 'test.jsonl',
                                'method_params': {'param1': 'value1'}
                            }
                        )

        # 验证结果
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.layer_scores, [{'name': 'layer1', 'score': 1.0}])
        self.assertEqual(result.method, 'std')
        self.assertEqual(result.patterns, ['*'])

        # 验证调用
        mock_prep_data.assert_called_once_with(mock_model, 'test.jsonl')
        mock_get_layers.assert_called_once_with(mock_model.model, ['*'])
        mock_run_analysis.assert_called_once()
        mock_factory.create_method.assert_called_once_with('std', param1='value1')

    def test_run_analysis_with_unsupported_data_type(self):
        """测试_run_analysis方法使用不支持的数据类型"""
        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        # 创建模拟模型
        mock_model = MagicMock(spec=nn.Module)
        mock_layer = MagicMock(spec=nn.Linear)
        mock_model.named_modules.return_value = [('layer1', mock_layer)]

        # 创建模拟分析方法
        mock_analysis_method = MagicMock()

        # 创建不支持的数据类型
        calib_data = ["unsupported_string_data"]

        # 执行测试，期望抛出异常
        with self.assertRaises(NotImplementedError):
            service._run_analysis(mock_model, ['layer1'], mock_analysis_method, calib_data)

    def test_run_analysis_no_target_layers_found(self):
        """测试_run_analysis方法没有找到目标层"""

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        # 创建模拟模型，只有非Linear层
        mock_model = MagicMock(spec=nn.Module)
        mock_layer = MagicMock(spec=nn.Conv2d)  # 非Linear层
        mock_model.named_modules.return_value = [('layer1', mock_layer)]

        mock_analysis_method = MagicMock()

        result = service._run_analysis(mock_model, ['layer1'], mock_analysis_method, None)

        # 验证结果为空
        self.assertEqual(result, [])


class TestAnalysisResult(TestComprehensiveAnalysisCoverage):
    """测试AnalysisResult类"""

    def test_analysis_result_init(self):
        """测试AnalysisResult初始化"""

        layer_scores = [{'name': 'layer1', 'score': 2.0}, {'name': 'layer2', 'score': 1.0}]
        result = AnalysisResult(layer_scores, 'test_method', ['*'])

        self.assertEqual(result.layer_scores, layer_scores)
        self.assertEqual(result.method, 'test_method')
        self.assertEqual(result.patterns, ['*'])

    def test_get_sorted_layers(self):
        """测试get_sorted_layers方法"""

        layer_scores = [
            {'name': 'layer1', 'score': 2.0},
            {'name': 'layer2', 'score': 1.0},
            {'name': 'layer3', 'score': 3.0}
        ]
        result = AnalysisResult(layer_scores, 'test_method', ['*'])

        # 测试升序排序
        sorted_layers = result.get_sorted_layers(reverse=False)
        self.assertEqual(sorted_layers[0]['name'], 'layer2')
        self.assertEqual(sorted_layers[1]['name'], 'layer1')
        self.assertEqual(sorted_layers[2]['name'], 'layer3')

        # 测试降序排序（默认）
        sorted_layers = result.get_sorted_layers(reverse=True)
        self.assertEqual(sorted_layers[0]['name'], 'layer3')
        self.assertEqual(sorted_layers[1]['name'], 'layer1')
        self.assertEqual(sorted_layers[2]['name'], 'layer2')


def create_mock_analysis_result(layer_scores: list) -> MagicMock:
    """
    构建模拟的 AnalysisResult 对象，用于测试输入

    Args:
        layer_scores: 层分数列表，每个元素为 {'name': 层名, 'score': 敏感度分数}

    Returns:
        MagicMock: 模拟的 AnalysisResult 实例
    """
    mock_result = MagicMock(spec=AnalysisResult)
    # 模拟 get_sorted_layers 方法（按 score 降序排序）
    mock_result.get_sorted_layers = MagicMock(return_value=sorted(layer_scores, key=lambda x: x['score'], reverse=True))
    # 模拟属性
    mock_result.method = "kurtosis"  # 模拟峰度算法（可替换为其他方法）
    mock_result.patterns = ["conv2d", "linear", "mlp"]  # 模拟分析的模式
    mock_result.layer_scores = layer_scores  # 模拟总层分数列表
    return mock_result


class TestPrintAnalysisResults(unittest.TestCase):
    """测试 _print_analysis_results 方法的单元测试类"""

    def test_normal_case_with_disable_level(self):
        """
        正常场景：指定合理的 disable_level，输出对应数量的敏感层
        验证点：基础信息打印、指定数量的层输出、YAML 格式正确性
        """
        # 1. 准备测试数据：5个层，分数从高到低排序
        test_layers = [
            {"name": "model.layers.26.mlp.down_proj", "score": 98.7654},
            {"name": "model.layers.4.mlp.down_proj", "score": 87.6543},
            {"name": "model.layers.1.mlp.down_proj", "score": 76.5432},
            {"name": "model.layers.3.mlp.down_proj", "score": 65.4321},
            {"name": "model.layers.2.mlp.down_proj", "score": 54.3210}
        ]
        mock_result = create_mock_analysis_result(test_layers)
        # 创建mock dataset_loader，因为LayerSelectorAnalysisService需要它
        mock_dataset_loader = MagicMock()
        test_instance = LayerSelectorAnalysisService(mock_dataset_loader)  # 初始化被测试类实例
        disable_level = 3  # 期望输出前3个敏感层

        # 2. 创建mock logger
        mock_logger = MagicMock()

        # 我们需要patch get_logger函数，因为@logger_setter装饰器会调用它
        with patch('msmodelslim.app.analysis_service.layer_selector.get_logger') as mock_get_logger:
            mock_get_logger.return_value = mock_logger

            # 3. 执行被测试方法
            test_instance._print_analysis_results(result=mock_result, disable_level=disable_level)

            # 4. 验证基本功能
            self.assertTrue(mock_logger.info.called, "mock_logger应该被调用")
            self.assertGreater(mock_logger.info.call_count, 5, "应该有多次日志调用")

            # 5. 验证一些关键的日志内容
            log_messages = [str(call[0][0]) if call[0] else "" for call in mock_logger.info.call_args_list]

            # 验证包含关键信息
            self.assertTrue(any("Layer Analysis Results" in str(msg) for msg in log_messages), "应该包含分析结果标题")
            self.assertTrue(any("kurtosis method" in str(msg) for msg in log_messages), "应该包含分析方法")
            self.assertTrue(any("conv2d" in str(msg) for msg in log_messages), "应该包含模式信息")
            self.assertTrue(any("Total layers analyzed: 5" in str(msg) for msg in log_messages), "应该包含总层数信息")


if __name__ == '__main__':
    unittest.main()
