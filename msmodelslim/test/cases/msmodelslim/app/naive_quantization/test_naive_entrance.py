"""
naive_entrance.py 的单元测试模块

本模块包含对 NaiveEntrance 类的全面单元测试，覆盖以下场景：
1. 类的初始化过程
2. run_quantization 方法的正常执行流程
3. run_quantization 方法的异常处理
4. 各种边界条件和错误情况
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import importlib


class TestNaiveEntrance:
    """NaiveEntrance 类的单元测试类"""

    def setup_method(self):
        """测试前准备：创建mock并保存原始模块"""
        # 需要mock的模块列表
        self.mock_modules = [
            'tqdm',
            'ascend_utils',
            'ascend_utils.common',
            'ascend_utils.common.security',
            'msmodelslim.infra',
            'msmodelslim.infra.practice_manager',
            'msmodelslim.tools',
            'msmodelslim.tools.logger',
            'msmodelslim.app.naive_quantization.quantization'
        ]

        # 保存原始模块引用
        self.original_modules = {}
        for module_name in self.mock_modules:
            self.original_modules[module_name] = sys.modules.get(module_name)
            sys.modules[module_name] = Mock()

        # 动态导入被测类（在mock之后）
        self.naive_entrance_module = importlib.import_module(
            'msmodelslim.app.naive_quantization.naive_entrance'
        )
        self.NaiveEntrance = self.naive_entrance_module.NaiveEntrance

    def teardown_method(self):
        """测试后清理：恢复原始模块"""
        for module_name, original_module in self.original_modules.items():
            if original_module is not None:
                sys.modules[module_name] = original_module
            else:
                # 如果原始模块不存在，则从sys.modules中移除
                sys.modules.pop(module_name, None)

        # 清除已导入的模块，避免影响其他测试
        if 'msmodelslim.app.naive_quantization.naive_entrance' in sys.modules:
            del sys.modules['msmodelslim.app.naive_quantization.naive_entrance']

    @patch('msmodelslim.app.naive_quantization.naive_entrance.get_valid_read_path')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.NaiveQuantization')
    def test_init_success(self, mock_naive_quantization, mock_get_valid_read_path):
        """
        测试 NaiveEntrance 类的成功初始化
        
        验证点：
        1. config_dir 路径计算正确
        2. get_valid_read_path 被正确调用
        3. NaiveQuantization 实例被正确创建
        """
        # 准备测试数据
        mock_practice_lab_dir = '/test/practice_lab'
        mock_get_valid_read_path.return_value = mock_practice_lab_dir
        mock_quantizer_instance = Mock()
        mock_naive_quantization.return_value = mock_quantizer_instance

        # 执行测试
        entrance = self.NaiveEntrance()

        # 验证结果
        # 验证 get_valid_read_path 被正确调用
        mock_get_valid_read_path.assert_called_once()
        call_args = mock_get_valid_read_path.call_args
        assert call_args[1]['is_dir'] is True
        
        # 验证 config_dir 被正确设置
        assert entrance.config_dir == Path(mock_practice_lab_dir)
        
        # 验证 NaiveQuantization 被正确实例化
        mock_naive_quantization.assert_called_once_with(Path(mock_practice_lab_dir))
        assert entrance.naive_quantizer == mock_quantizer_instance

    @patch('msmodelslim.app.naive_quantization.naive_entrance.get_valid_read_path')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.NaiveQuantization')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.quant_backend')
    def test_run_quantization_success(self, mock_quant_backend, mock_naive_quantization, mock_get_valid_read_path):
        """
        测试 run_quantization 方法成功执行的情况
        
        验证点：
        1. get_best_practice 方法被正确调用并传递参数
        2. quant_backend 实例被创建并调用 quant_process
        3. 返回正确的 best_config
        """
        # 准备测试数据
        mock_get_valid_read_path.return_value = '/test/practice_lab'
        mock_quantizer_instance = Mock()
        mock_naive_quantization.return_value = mock_quantizer_instance
        
        # 模拟 get_best_practice 返回的配置
        mock_best_config = {
            'model_type': 'bert',
            'quant_type': 'int8',
            'device': 'gpu',
            'batch_size': 32
        }
        mock_quantizer_instance.get_best_practice.return_value = mock_best_config
        
        # 模拟 quant_backend 实例
        mock_quant_instance = Mock()
        mock_quant_backend.return_value = mock_quant_instance
        
        # 模拟 args 参数
        mock_args = Mock()
        mock_args.model_type = 'bert'
        mock_args.config_path = '/path/to/config'
        mock_args.quant_type = 'int8'
        mock_args.device = 'gpu'
        mock_args.model_path = '/path/to/model'
        mock_args.save_path = '/path/to/save'
        mock_args.trust_remote_code = True

        # 执行测试
        entrance = self.NaiveEntrance()
        result = entrance.run_quantization(mock_args)

        # 验证结果
        # 验证 get_best_practice 被正确调用
        mock_quantizer_instance.get_best_practice.assert_called_once_with(
            model_type=mock_args.model_type,
            config_path=mock_args.config_path,
            quant_type=mock_args.quant_type,
            device=mock_args.device,
            model_path=mock_args.model_path,
            save_path=mock_args.save_path,
            trust_remote_code=mock_args.trust_remote_code
        )
        
        # 验证 quant_backend 被实例化
        mock_quant_backend.assert_called_once()
        
        # 验证 quant_process 被调用
        mock_quant_instance.quant_process.assert_called_once_with(mock_best_config)
        
        # 验证返回值
        assert result == mock_best_config

    @patch('msmodelslim.app.naive_quantization.naive_entrance.get_valid_read_path')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.NaiveQuantization')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.quant_backend')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.msmodelslim_logger.logger_error')
    def test_run_quantization_value_error(self, mock_logger_error, mock_quant_backend, 
                                        mock_naive_quantization, mock_get_valid_read_path):
        """
        测试 run_quantization 方法遇到 ValueError 异常的情况
        
        验证点：
        1. ValueError 异常被正确捕获
        2. 错误日志被正确记录
        3. 返回 None
        """
        # 准备测试数据
        mock_get_valid_read_path.return_value = '/test/practice_lab'
        mock_quantizer_instance = Mock()
        mock_naive_quantization.return_value = mock_quantizer_instance
        
        # 模拟 get_best_practice 抛出 ValueError
        error_message = "Invalid model type"
        mock_quantizer_instance.get_best_practice.side_effect = ValueError(error_message)
        
        # 模拟 args 参数
        mock_args = Mock()
        mock_args.model_type = 'invalid_type'
        mock_args.config_path = '/path/to/config'
        mock_args.quant_type = 'int8'
        mock_args.device = 'gpu'
        mock_args.model_path = '/path/to/model'
        mock_args.save_path = '/path/to/save'
        mock_args.trust_remote_code = False

        # 执行测试
        entrance = self.NaiveEntrance()
        result = entrance.run_quantization(mock_args)

        # 验证结果
        # 验证 get_best_practice 被调用
        mock_quantizer_instance.get_best_practice.assert_called_once()
        
        # 验证错误日志被记录
        mock_logger_error.assert_called_once_with(f"Error: {error_message}")
        
        # 验证 quant_backend 未被调用（因为异常提前返回）
        mock_quant_backend.assert_not_called()
        
        # 验证返回 None
        assert result is None

    @patch('msmodelslim.app.naive_quantization.naive_entrance.get_valid_read_path')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.NaiveQuantization')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.quant_backend')
    def test_run_quantization_quant_process_exception(self, mock_quant_backend, 
                                                    mock_naive_quantization, mock_get_valid_read_path):
        """
        测试 run_quantization 方法中 quant_process 抛出非 ValueError 异常的情况
        
        验证点：
        1. 非 ValueError 异常不被捕获，正常向上抛出
        2. get_best_practice 正常执行
        3. quant_process 被调用但抛出异常
        """
        # 准备测试数据
        mock_get_valid_read_path.return_value = '/test/practice_lab'
        mock_quantizer_instance = Mock()
        mock_naive_quantization.return_value = mock_quantizer_instance
        
        # 模拟正常的 get_best_practice 返回
        mock_best_config = {'model_type': 'bert'}
        mock_quantizer_instance.get_best_practice.return_value = mock_best_config
        
        # 模拟 quant_process 抛出 RuntimeError
        mock_quant_instance = Mock()
        mock_quant_instance.quant_process.side_effect = RuntimeError("Quantization failed")
        mock_quant_backend.return_value = mock_quant_instance
        
        # 模拟 args 参数
        mock_args = Mock()
        mock_args.model_type = 'bert'
        mock_args.config_path = '/path/to/config'
        mock_args.quant_type = 'int8'
        mock_args.device = 'gpu'
        mock_args.model_path = '/path/to/model'
        mock_args.save_path = '/path/to/save'
        mock_args.trust_remote_code = True

        # 执行测试并验证异常
        entrance = self.NaiveEntrance()
        with pytest.raises(RuntimeError, match="Quantization failed"):
            entrance.run_quantization(mock_args)

        # 验证 get_best_practice 被调用
        mock_quantizer_instance.get_best_practice.assert_called_once()
        
        # 验证 quant_process 被调用
        mock_quant_instance.quant_process.assert_called_once_with(mock_best_config)

    @patch('msmodelslim.app.naive_quantization.naive_entrance.get_valid_read_path')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.NaiveQuantization')
    def test_run_quantization_with_different_args(self, mock_naive_quantization, mock_get_valid_read_path):
        """
        测试 run_quantization 方法使用不同参数的情况
        
        验证点：
        1. 不同的参数值被正确传递给 get_best_practice
        2. 参数映射关系正确（特别是 device 到 device_type 的映射）
        """
        # 准备测试数据
        mock_get_valid_read_path.return_value = '/test/practice_lab'
        mock_quantizer_instance = Mock()
        mock_naive_quantization.return_value = mock_quantizer_instance
        
        # 模拟不同的 args 参数
        test_cases = [
            {
                'model_type': 'gpt2',
                'config_path': '/custom/config',
                'quant_type': 'fp16',
                'device': 'cpu',
                'model_path': '/custom/model',
                'save_path': '/custom/save',
                'trust_remote_code': False
            },
            {
                'model_type': 'resnet',
                'config_path': None,
                'quant_type': 'int4',
                'device': 'npu',
                'model_path': '/another/model',
                'save_path': '/another/save',
                'trust_remote_code': True
            }
        ]

        entrance = self.NaiveEntrance()
        
        for i, test_case in enumerate(test_cases):
            # 重置 mock
            mock_quantizer_instance.reset_mock()
            mock_quantizer_instance.get_best_practice.return_value = {'config': f'test_{i}'}
            
            # 创建 mock args
            mock_args = Mock()
            for key, value in test_case.items():
                setattr(mock_args, key, value)

            # 执行测试
            with patch('msmodelslim.app.naive_quantization.naive_entrance.quant_backend') as mock_quant_backend:
                mock_quant_instance = Mock()
                mock_quant_backend.return_value = mock_quant_instance
                
                result = entrance.run_quantization(mock_args)

                # 验证参数传递
                mock_quantizer_instance.get_best_practice.assert_called_once_with(
                    model_type=test_case['model_type'],
                    config_path=test_case['config_path'],
                    quant_type=test_case['quant_type'],
                    device=test_case['device'],
                    model_path=test_case['model_path'],
                    save_path=test_case['save_path'],
                    trust_remote_code=test_case['trust_remote_code']
                )

    @patch('msmodelslim.app.naive_quantization.naive_entrance.os.path.dirname')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.os.path.abspath')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.os.path.join')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.get_valid_read_path')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.NaiveQuantization')
    def test_init_path_computation(self, mock_naive_quantization, mock_get_valid_read_path,
                                  mock_join, mock_abspath, mock_dirname):
        """
        测试初始化过程中路径计算的正确性
        
        验证点：
        1. os.path 相关函数的调用顺序和参数
        2. practice_lab_dir 路径的正确构建
        """
        # 准备测试数据
        mock_dirname.return_value = '/current/dir'
        mock_join.return_value = '/current/dir/../../practice_lab'
        mock_abspath.side_effect = ['/current/file/path', '/absolute/practice_lab']
        mock_get_valid_read_path.return_value = '/validated/practice_lab'
        mock_quantizer_instance = Mock()
        mock_naive_quantization.return_value = mock_quantizer_instance

        # 执行测试
        entrance = self.NaiveEntrance()

        # 验证路径计算过程
        # 验证 dirname 和第一次 abspath 被调用
        assert mock_dirname.call_count == 1
        assert mock_abspath.call_count == 2
        
        # 验证 join 被调用用于构建相对路径
        mock_join.assert_called_once_with('/current/dir', '../../practice_lab')
        
        # 验证 get_valid_read_path 被调用验证路径
        mock_get_valid_read_path.assert_called_once_with('/absolute/practice_lab', is_dir=True)

    def test_class_attributes_after_init(self):
        """
        测试初始化后类属性的完整性
        
        验证点：
        1. 所有必要的属性都被正确设置
        2. 属性类型正确
        """
        with patch('msmodelslim.app.naive_quantization.naive_entrance.get_valid_read_path') as mock_get_valid_read_path, \
             patch('msmodelslim.app.naive_quantization.naive_entrance.NaiveQuantization') as mock_naive_quantization:
            
            mock_get_valid_read_path.return_value = '/test/practice_lab'
            mock_quantizer_instance = Mock()
            mock_naive_quantization.return_value = mock_quantizer_instance

            entrance = self.NaiveEntrance()

            # 验证属性存在性和类型
            assert hasattr(entrance, 'config_dir')
            assert hasattr(entrance, 'naive_quantizer')
            assert isinstance(entrance.config_dir, Path)
            assert entrance.naive_quantizer == mock_quantizer_instance

    @patch('msmodelslim.app.naive_quantization.naive_entrance.get_valid_read_path')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.NaiveQuantization')
    @patch('msmodelslim.app.naive_quantization.naive_entrance.quant_backend')
    def test_complete_workflow(self, mock_quant_backend, mock_naive_quantization, mock_get_valid_read_path):
        """
        测试完整的工作流程
        
        验证点：
        1. 从初始化到量化完成的完整流程
        2. 各个组件之间的交互
        """
        # 准备测试数据
        mock_get_valid_read_path.return_value = '/test/practice_lab'
        mock_quantizer_instance = Mock()
        mock_naive_quantization.return_value = mock_quantizer_instance
        
        mock_best_config = {
            'model_type': 'bert',
            'precision': 'int8',
            'optimization_level': 'O2'
        }
        mock_quantizer_instance.get_best_practice.return_value = mock_best_config
        
        mock_quant_instance = Mock()
        mock_quant_backend.return_value = mock_quant_instance
        
        # 创建测试参数
        class TestArgs:
            def __init__(self):
                self.model_type = 'bert'
                self.config_path = '/test/config.yaml'
                self.quant_type = 'int8'
                self.device = 'gpu'
                self.model_path = '/test/model.pth'
                self.save_path = '/test/output'
                self.trust_remote_code = True

        test_args = TestArgs()

        # 执行完整流程
        entrance = self.NaiveEntrance()
        result = entrance.run_quantization(test_args)

        # 验证完整流程
        # 1. 初始化阶段
        mock_get_valid_read_path.assert_called_once()
        mock_naive_quantization.assert_called_once()
        
        # 2. 量化阶段
        mock_quantizer_instance.get_best_practice.assert_called_once()
        mock_quant_backend.assert_called_once()
        mock_quant_instance.quant_process.assert_called_once_with(mock_best_config)
        
        # 3. 返回结果
        assert result == mock_best_config


if __name__ == '__main__':
    """
    运行测试的入口点
    
    使用方法：
    1. 直接运行此文件：python test_naive_entrance.py
    2. 使用 pytest：pytest test_naive_entrance.py
    3. 带覆盖率报告：pytest --cov=msmodelslim.app.naive_quantization.naive_entrance test_naive_entrance.py
    """
    pytest.main([__file__, '-v', '--tb=short'])
