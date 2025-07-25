# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
命令行入口模块的单元测试
本文件对 __main__.py 进行全面的单元测试，包括：
1. 参数解析和验证
2. 命令分发逻辑
3. 帮助信息生成
4. 错误处理
确保高代码覆盖率和测试质量。
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import argparse
from importlib import import_module


class TestMainCLI:
    """命令行入口测试类"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        # 保存原始模块引用，用于后续恢复
        self.original_modules = {}
        # 定义需要模拟的模块
        self.mock_modules = {
            'msmodelslim.cli.naive_quant.naive_quant': MagicMock(),
            'msmodelslim.infra.practice_manager': MagicMock(),
            'msmodelslim.utils.safe_utils': MagicMock(),
        }

        # 应用所有模拟并保存原始模块
        for module_name, mock_module in self.mock_modules.items():
            self.original_modules[module_name] = sys.modules.get(module_name)
            sys.modules[module_name] = mock_module

        # 设置模拟模块的属性
        self.mock_modules['msmodelslim.infra.practice_manager'].SUPPRORTED_QUANT_TYPES = ['w8a8', 'w4a8']
        self.mock_modules['msmodelslim.utils.safe_utils'].cmd_bool = lambda x: x.lower() == 'true'

        # 重置所有mock
        for mock_module in self.mock_modules.values():
            mock_module.reset_mock()

        # 保存原始sys.argv
        self.original_argv = sys.argv.copy()

        # 动态导入被测模块，确保 coverage 能跟踪
        self.main_module = import_module('msmodelslim.cli.__main__')

        global main, MIND_STUDIO_LOGO, FAQ_HOME
        main = self.main_module.main
        MIND_STUDIO_LOGO = self.main_module.MIND_STUDIO_LOGO
        FAQ_HOME = self.main_module.FAQ_HOME

    def teardown_method(self):
        """每个测试方法执行后的清理"""
        # 恢复原始sys.argv
        sys.argv = self.original_argv

        # 恢复原始模块，避免影响其他测试
        for module_name, original_module in self.original_modules.items():
            if original_module is not None:
                sys.modules[module_name] = original_module
            else:
                # 如果原始模块不存在，则从sys.modules中移除
                del sys.modules[module_name]

        # 清理测试导入的模块（防止影响其他测试）
        if 'msmodelslim.cli.__main__' in sys.modules:
            del sys.modules['msmodelslim.cli.__main__']

        # 清理全局变量
        global main, MIND_STUDIO_LOGO, FAQ_HOME
        main = None
        MIND_STUDIO_LOGO = None
        FAQ_HOME = None
    
    def test_help_message(self, capsys):
        """测试帮助信息输出
        
        验证：
        1. 帮助信息包含程序名称
        2. 帮助信息包含MindStudio标识
        3. 帮助信息包含FAQ链接
        4. 帮助信息包含可用命令列表
        """
        sys.argv = ['msmodelslim', '--help']
        with pytest.raises(SystemExit):
            self.main_module.main()
        captured = capsys.readouterr()
        assert "MsModelSlim" in captured.out
        assert MIND_STUDIO_LOGO in captured.out
        assert FAQ_HOME in captured.out
        assert "Available commands" in captured.out
        assert "quant" in captured.out  # 验证子命令显示
    
    def test_no_command(self, capsys):
        """测试无命令参数的情况
        
        验证：
        1. 无命令时显示帮助信息
        2. 程序正常返回
        """
        sys.argv = ['msmodelslim']
        self.main_module.main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out
    
    def test_quant_command_missing_required_args(self):
        """测试量化命令缺少必需参数的情况
        
        验证：
        1. 缺少model_type时抛出异常
        2. 缺少model_path时抛出异常
        3. 缺少save_path时抛出异常
        """
        # 测试缺少model_type
        sys.argv = ['msmodelslim', 'quant', '--model_path', '/path/to/model', '--save_path', '/path/to/save']
        with pytest.raises(SystemExit):
            self.main_module.main()
        
        # 测试缺少model_path
        sys.argv = ['msmodelslim', 'quant', '--model_type', 'Qwen2.5-7B-Instruct', '--save_path', '/path/to/save']
        with pytest.raises(SystemExit):
            self.main_module.main()
        
        # 测试缺少save_path
        sys.argv = ['msmodelslim', 'quant', '--model_type', 'Qwen2.5-7B-Instruct', '--model_path', '/path/to/model']
        with pytest.raises(SystemExit):
            self.main_module.main()
    
    def test_quant_command_valid_args(self):
        """测试量化命令有效参数的情况
        
        验证：
        1. 所有必需参数都提供时正常执行
        2. 默认参数值正确
        3. 命令正确分发到quant_main
        """
        mock_quant = self.mock_modules['msmodelslim.cli.naive_quant.naive_quant']
        sys.argv = ['msmodelslim', 'quant', 
                   '--model_type', 'Qwen2.5-7B-Instruct',
                   '--model_path', '/path/to/model',
                   '--save_path', '/path/to/save']
        self.main_module.main()
        
        # 验证quant_main被调用
        mock_quant.main.assert_called_once()
        
        # 验证参数正确传递
        args = mock_quant.main.call_args[0][0]
        assert args.model_type == 'Qwen2.5-7B-Instruct'
        assert args.model_path == '/path/to/model'
        assert args.save_path == '/path/to/save'
        assert args.device == 'npu'  # 验证默认值
        assert args.trust_remote_code is False  # 验证默认值
    
    def test_quant_command_device_validation(self):
        """测试设备类型参数验证
        
        验证：
        1. 支持npu和cpu设备
        2. 无效设备类型时抛出异常
        """
        mock_quant = self.mock_modules['msmodelslim.cli.naive_quant.naive_quant']
        
        # 测试有效设备类型
        sys.argv = ['msmodelslim', 'quant',
                   '--model_type', 'Qwen2.5-7B-Instruct',
                   '--model_path', '/path/to/model',
                   '--save_path', '/path/to/save',
                   '--device', 'cpu']
        self.main_module.main()
        args = mock_quant.main.call_args[0][0]
        assert args.device == 'cpu'
        
        # 测试无效设备类型
        sys.argv = ['msmodelslim', 'quant',
                   '--model_type', 'Qwen2.5-7B-Instruct',
                   '--model_path', '/path/to/model',
                   '--save_path', '/path/to/save',
                   '--device', 'invalid_device']
        with pytest.raises(SystemExit):
            self.main_module.main()
    
    def test_quant_command_trust_remote_code(self):
        """测试trust_remote_code参数处理
        
        验证：
        1. 布尔值参数正确解析
        2. 默认值正确
        3. 不同布尔值表示方式都能正确解析
        """
        mock_quant = self.mock_modules['msmodelslim.cli.naive_quant.naive_quant']
        
        # 测试True值
        sys.argv = ['msmodelslim', 'quant',
                   '--model_type', 'Qwen2.5-7B-Instruct',
                   '--model_path', '/path/to/model',
                   '--save_path', '/path/to/save',
                   '--trust_remote_code', 'True']
        self.main_module.main()
        args = mock_quant.main.call_args[0][0]
        assert args.trust_remote_code is True
        
        # 测试False值
        sys.argv = ['msmodelslim', 'quant',
                   '--model_type', 'Qwen2.5-7B-Instruct',
                   '--model_path', '/path/to/model',
                   '--save_path', '/path/to/save',
                   '--trust_remote_code', 'False']
        self.main_module.main()
        args = mock_quant.main.call_args[0][0]
        assert args.trust_remote_code is False
    
    def test_quant_type_validation(self):
        """测试量化类型参数验证
        
        验证：
        1. 支持的量化类型可以正常使用
        2. 不支持的量化类型会抛出异常
        """
        mock_quant = self.mock_modules['msmodelslim.cli.naive_quant.naive_quant']
        
        # 测试支持的量化类型
        sys.argv = ['msmodelslim', 'quant',
                   '--model_type', 'Qwen2.5-7B-Instruct',
                   '--model_path', '/path/to/model',
                   '--save_path', '/path/to/save',
                   '--quant_type', 'w8a8']
        self.main_module.main()
        args = mock_quant.main.call_args[0][0]
        assert args.quant_type == 'w8a8'
        
        # 测试不支持的量化类型
        sys.argv = ['msmodelslim', 'quant',
                   '--model_type', 'Qwen2.5-7B-Instruct',
                   '--model_path', '/path/to/model',
                   '--save_path', '/path/to/save',
                   '--quant_type', 'invalid_type']
        with pytest.raises(SystemExit):
            self.main_module.main()
    
    def test_config_path_optional(self):
        """测试可选配置路径参数
        
        验证：
        1. 配置路径参数是可选的
        2. 提供配置路径时正确传递
        """
        mock_quant = self.mock_modules['msmodelslim.cli.naive_quant.naive_quant']
        
        sys.argv = ['msmodelslim', 'quant',
                   '--model_type', 'Qwen2.5-7B-Instruct',
                   '--model_path', '/path/to/model',
                   '--save_path', '/path/to/save',
                   '--config_path', '/path/to/config']
        self.main_module.main()
        args = mock_quant.main.call_args[0][0]
        assert args.config_path == '/path/to/config'
    
    def test_unknown_command(self, capsys):
        """测试未知命令处理
        
        验证：
        1. 未知命令时显示帮助信息
        2. 程序正常退出
        """
        sys.argv = ['msmodelslim', 'unknown_command']
        with pytest.raises(SystemExit) as exc_info:
            self.main_module.main()
        assert exc_info.value.code == 2  # 验证退出码
        captured = capsys.readouterr()
        assert "usage:" in captured.err  # 错误信息输出到stderr
        assert "invalid choice" in captured.err


if __name__ == '__main__':
    """
    测试运行入口点
    
    使用方法：
    1. 直接运行：python test_main.py
    2. 使用pytest：pytest test_main.py -v
    3. 检查覆盖率：pytest test_main.py --cov=msmodelslim.cli.__main__
    """
    pytest.main([__file__, '-v'])
