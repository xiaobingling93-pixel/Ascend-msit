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

from unittest.mock import Mock, patch, call

import pytest

from msserviceprofiler.vllm_profiler.symbol_watcher import (
    SymbolWatchFinder, make_default_time_hook, register_dynamic_hook
)


@pytest.fixture
def symbol_watch_finder():
    """提供 SymbolWatchFinder 实例的 fixture"""
    return SymbolWatchFinder()


@pytest.fixture
def sample_config():
    """提供示例配置的 fixture"""
    return [
        {'symbol': 'module1:Class1.method1', 'handler': 'handlers:time_hook'},
        {'symbol': 'module2:function2', 'domain': 'Test', 'attributes': {'key': 'value'}},
        {'symbol': 'parent.child.grandchild:function3', 'name': 'GrandchildFunction'}
    ]


@pytest.fixture
def mock_loader():
    """提供模拟加载器的 fixture"""
    mock_loader = Mock()
    mock_loader._vllm_profiler_wrapped = False
    mock_loader.create_module.return_value = None
    return mock_loader


@pytest.fixture
def mock_spec(mock_loader):
    """提供模拟模块规范的 fixture"""
    mock_spec = Mock()
    mock_spec.loader = mock_loader
    return mock_spec


class TestSymbolWatchFinderInitialization:
    """测试 SymbolWatchFinder 初始化"""
    
    @staticmethod
    def test_initialization(symbol_watch_finder):
        """测试初始化状态"""
        assert symbol_watch_finder._symbol_hooks == {}
        assert symbol_watch_finder._config_loaded is False
        assert symbol_watch_finder._applied_hooks == set()


class TestLoadSymbolConfig:
    """测试 load_symbol_config 方法"""
    
    @staticmethod
    def test_load_symbol_config(symbol_watch_finder, sample_config):
        """测试加载符号配置"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        assert symbol_watch_finder._config_loaded is True
        assert len(symbol_watch_finder._symbol_hooks) == 3
        assert 'symbol_0' in symbol_watch_finder._symbol_hooks
        assert 'symbol_1' in symbol_watch_finder._symbol_hooks
        assert symbol_watch_finder._symbol_hooks['symbol_0']['symbol'] == 'module1:Class1.method1'

    @staticmethod
    def test_load_symbol_config_empty(symbol_watch_finder):
        """测试加载空配置"""
        symbol_watch_finder.load_symbol_config([])
        assert symbol_watch_finder._config_loaded is True
        assert symbol_watch_finder._symbol_hooks == {}


class TestIsTargetSymbol:
    """测试 _is_target_symbol 方法"""
    
    @staticmethod
    def test_is_target_symbol_not_loaded(symbol_watch_finder):
        """测试未加载配置时的目标符号检查"""
        result = symbol_watch_finder._is_target_symbol('some.module')
        assert result is False

    @staticmethod
    def test_is_target_symbol_direct_match(symbol_watch_finder, sample_config):
        """测试直接模块匹配"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        result = symbol_watch_finder._is_target_symbol('module1')
        assert result is True

    @staticmethod
    def test_is_target_symbol_parent_package_match(symbol_watch_finder, sample_config):
        """测试父包匹配"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        result = symbol_watch_finder._is_target_symbol('parent.child')
        assert result is True

    @staticmethod
    def test_is_target_symbol_no_match(symbol_watch_finder, sample_config):
        """测试无匹配情况"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        result = symbol_watch_finder._is_target_symbol('unrelated.module')
        assert result is False


class TestFindSpec:
    """测试 find_spec 方法"""
    
    @staticmethod
    @pytest.mark.parametrize("module_name,expected_call", [
        ('unrelated.module', False),
        ('module1', True),
        ('module2', True)
    ])
    def test_find_spec_various_modules(symbol_watch_finder, sample_config, module_name, expected_call, mock_spec):
        """测试各种模块的查找规范"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        with patch('importlib.machinery.PathFinder.find_spec', return_value=mock_spec) as mock_find:
            result = symbol_watch_finder.find_spec(module_name, None)
            
            if expected_call:
                mock_find.assert_called_once_with(module_name, None)
            else:
                mock_find.assert_not_called()

    @staticmethod
    def test_find_spec_target_module_no_spec(symbol_watch_finder, sample_config):
        """测试目标模块但找不到规范的情况"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        with patch('importlib.machinery.PathFinder.find_spec', return_value=None):
            result = symbol_watch_finder.find_spec('module1', None)
            assert result is None

    @staticmethod
    def test_find_spec_target_module_no_loader(symbol_watch_finder, sample_config):
        """测试目标模块但规范无加载器的情况"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        mock_spec = Mock(loader=None)
        with patch('importlib.machinery.PathFinder.find_spec', return_value=mock_spec):
            result = symbol_watch_finder.find_spec('module1', None)
            assert result == mock_spec

    @staticmethod
    def test_find_spec_already_wrapped(symbol_watch_finder, sample_config, mock_loader):
        """测试已包装的加载器"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        mock_loader._vllm_profiler_wrapped = True
        mock_spec = Mock(loader=mock_loader)
        
        with patch('importlib.machinery.PathFinder.find_spec', return_value=mock_spec):
            result = symbol_watch_finder.find_spec('module1', None)
            assert result == mock_spec

    @staticmethod
    def test_find_spec_successful_wrapping(symbol_watch_finder, sample_config, mock_loader, mock_spec):
        """测试成功包装加载器"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        with patch('importlib.machinery.PathFinder.find_spec', return_value=mock_spec):
            result = symbol_watch_finder.find_spec('module1', None)
            
            # 验证加载器被包装
            assert result.loader != mock_loader
            assert hasattr(result.loader, '_finder')
            assert result.loader._finder == symbol_watch_finder
            assert result.loader._vllm_profiler_wrapped is True


class TestParseSymbolPath:
    """测试 _parse_symbol_path 方法"""
    
    @staticmethod
    @pytest.mark.parametrize("symbol_path,expected", [
        ('module.path:ClassName.method_name', ('module.path', 'method_name', 'ClassName')),
        ('module.path:function_name', ('module.path', 'function_name', None)),
        ('pkg.mod:Cls.meth', ('pkg.mod', 'meth', 'Cls')),
    ])
    def test_parse_symbol_path(symbol_watch_finder, symbol_path, expected):
        """测试解析符号路径"""
        result = symbol_watch_finder._parse_symbol_path(symbol_path)
        assert result == expected


class TestCreateHandlerFunction:
    """测试 _create_handler_function 方法"""
    
    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.importlib.import_module')
    def test_create_handler_function_custom(mock_import_module, symbol_watch_finder):
        """测试创建自定义处理函数"""
        # 模拟导入的模块和函数
        mock_module = Mock()
        mock_handler = Mock()
        mock_import_module.return_value = mock_module
        mock_module.custom_handler = mock_handler
        
        symbol_info = {
            'symbol': 'module:function',
            'handler': 'custom.module:custom_handler'
        }
        
        result = symbol_watch_finder._create_handler_function(symbol_info, 'function')
        
        mock_import_module.assert_called_once_with('custom.module')
        assert result == mock_handler

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.make_default_time_hook')
    def test_create_handler_function_default(mock_make_default, symbol_watch_finder):
        """测试创建默认处理函数"""
        mock_default_handler = Mock()
        mock_make_default.return_value = mock_default_handler
        
        symbol_info = {
            'symbol': 'module:function',
            'domain': 'TestDomain',
            'name': 'TestName',
            'attributes': {'key': 'value'}
        }
        
        result = symbol_watch_finder._create_handler_function(symbol_info, 'function')
        
        mock_make_default.assert_called_once_with(
            domain='TestDomain',
            name='TestName',
            attributes={'key': 'value'}
        )
        assert result == mock_default_handler

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.make_default_time_hook')
    def test_create_handler_function_minimal_args(mock_make_default, symbol_watch_finder):
        """测试创建默认处理函数（最小参数）"""
        mock_default_handler = Mock()
        mock_make_default.return_value = mock_default_handler
        
        symbol_info = {
            'symbol': 'module:function'
            # 没有 domain, name, attributes
        }
        
        result = symbol_watch_finder._create_handler_function(symbol_info, 'function')
        
        mock_make_default.assert_called_once_with(
            domain="Default",
            name="function",
            attributes=None
        )
        assert result == mock_default_handler


class TestBuildHookPoints:
    """测试 _build_hook_points 方法"""
    
    @staticmethod
    @pytest.mark.parametrize("module_path,method_name,class_name,expected", [
        ('test.module', 'test_method', 'TestClass', [('test.module', 'TestClass.test_method')]),
        ('test.module', 'test_method', None, [('test.module', 'test_method')]),
        ('pkg.mod', 'func', 'Cls', [('pkg.mod', 'Cls.func')]),
    ])
    def test_build_hook_points(symbol_watch_finder, module_path, method_name, class_name, expected):
        """测试构建钩子点"""
        result = symbol_watch_finder._build_hook_points(module_path, method_name, class_name)
        assert result == expected


class TestRegisterAndApplyHook:
    """测试 _register_and_apply_hook 方法"""
    
    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.register_dynamic_hook')
    def test_register_and_apply_hook(mock_register, symbol_watch_finder):
        """测试注册和应用钩子"""
        mock_hooker = Mock()
        mock_register.return_value = mock_hooker
        
        symbol_info = {
            'min_version': '1.0',
            'max_version': '2.0',
            'caller_filter': lambda x: True
        }
        hook_points = [('module', 'hook_point')]
        handler_func = Mock()
        
        result = symbol_watch_finder._register_and_apply_hook(symbol_info, hook_points, handler_func)
        
        mock_register.assert_called_once_with(
            hook_list=hook_points,
            hook_func=handler_func,
            min_version='1.0',
            max_version='2.0',
            caller_filter=symbol_info['caller_filter']
        )
        mock_hooker.init.assert_called_once()
        assert result == mock_hooker

    @staticmethod
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.register_dynamic_hook')
    def test_register_and_apply_hook_minimal_args(mock_register, symbol_watch_finder):
        """测试注册和应用钩子（最小参数）"""
        mock_hooker = Mock()
        mock_register.return_value = mock_hooker
        
        symbol_info = {}  # 空配置
        hook_points = [('module', 'hook_point')]
        handler_func = Mock()
        
        result = symbol_watch_finder._register_and_apply_hook(symbol_info, hook_points, handler_func)
        
        mock_register.assert_called_once_with(
            hook_list=hook_points,
            hook_func=handler_func,
            min_version=None,
            max_version=None,
            caller_filter=None
        )
        assert result == mock_hooker


class TestApplySingleSymbolHook:
    """测试 _apply_single_symbol_hook 方法"""
    
    @staticmethod
    @patch.object(SymbolWatchFinder, '_register_and_apply_hook')
    @patch.object(SymbolWatchFinder, '_build_hook_points')
    @patch.object(SymbolWatchFinder, '_create_handler_function')
    @patch.object(SymbolWatchFinder, '_parse_symbol_path')
    def test_apply_single_symbol_hook_success(
        mock_parse, mock_create_handler, mock_build_hooks, mock_register, symbol_watch_finder
    ):
        """测试成功应用单个符号钩子"""
        # 设置模拟返回值
        mock_parse.return_value = ('module.path', 'method_name', 'ClassName')
        mock_handler = Mock()
        mock_create_handler.return_value = mock_handler
        mock_build_hooks.return_value = [('module.path', 'ClassName.method_name')]
        mock_hooker = Mock()
        mock_register.return_value = mock_hooker
        
        symbol_info = {'symbol': 'module.path:ClassName.method_name'}
        
        # 执行测试
        symbol_watch_finder._apply_single_symbol_hook('symbol_0', symbol_info)
        
        # 验证调用链
        mock_parse.assert_called_once_with('module.path:ClassName.method_name')
        mock_create_handler.assert_called_once_with(symbol_info, 'method_name')
        mock_build_hooks.assert_called_once_with('module.path', 'method_name', 'ClassName')
        mock_register.assert_called_once_with(symbol_info, [('module.path', 'ClassName.method_name')], mock_handler)
        
        # 验证已应用钩子记录
        assert 'module.path:ClassName.method_name' in symbol_watch_finder._applied_hooks

    @staticmethod
    def test_apply_single_symbol_hook_already_applied(symbol_watch_finder):
        """测试应用已存在的钩子（应跳过）"""
        symbol_info = {'symbol': 'module:function'}
        symbol_watch_finder._applied_hooks.add('module:function')
        
        # 使用 patch 来验证内部方法没有被调用
        with patch.object(symbol_watch_finder, '_parse_symbol_path') as mock_parse:
            symbol_watch_finder._apply_single_symbol_hook('symbol_0', symbol_info)
            mock_parse.assert_not_called()

    @staticmethod
    @patch.object(SymbolWatchFinder, '_parse_symbol_path')
    def test_apply_single_symbol_hook_exception(mock_parse, symbol_watch_finder):
        """测试应用钩子时出现异常"""
        mock_parse.side_effect = Exception("Test error")
        symbol_info = {'symbol': 'module:function'}
        
        # 应该捕获异常而不抛出
        symbol_watch_finder._apply_single_symbol_hook('symbol_0', symbol_info)


class TestApplySymbolHooksForModule:
    """测试 _apply_symbol_hooks_for_module 方法"""

    @staticmethod
    @patch.object(SymbolWatchFinder, '_apply_single_symbol_hook')
    def test_apply_symbol_hooks_for_module_success(mock_apply_single, symbol_watch_finder):
        """测试成功应用模块符号钩子"""
        # 确保正确设置 symbol_hooks
        config_data = [
            {'symbol': 'module:func1'},
            {'symbol': 'module:func2'}
        ]
        symbol_watch_finder.load_symbol_config(config_data)
        
        # 使用真实的 module_symbols 格式
        module_symbols = []
        for symbol_id, symbol_info in symbol_watch_finder._symbol_hooks.items():
            module_symbols.append((symbol_id, symbol_info))
        
        symbol_watch_finder._apply_symbol_hooks_for_module('test.module', module_symbols)
        
        assert mock_apply_single.call_count == 2
        
        # 检查调用参数
        calls = mock_apply_single.call_args_list
        assert len(calls) == 2
        
        # 验证每个调用的参数
        for _, call_args in enumerate(calls):
            symbol_id, symbol_info = call_args[0]  # 解包位置参数
            assert symbol_id.startswith('symbol_')
            assert 'symbol' in symbol_info
            assert symbol_info['symbol'].startswith('module:func')

    @staticmethod
    @patch.object(SymbolWatchFinder, '_apply_single_symbol_hook')
    def test_apply_symbol_hooks_for_module_exception(mock_apply_single, symbol_watch_finder):
        """测试应用模块符号钩子时出现异常"""
        mock_apply_single.side_effect = Exception("Test error")
        module_symbols = [('symbol_0', {'symbol': 'module:func'})]
        
        # 应该捕获异常而不抛出
        symbol_watch_finder._apply_symbol_hooks_for_module('test.module', module_symbols)


class TestOnSymbolModuleLoaded:
    """测试 _on_symbol_module_loaded 方法"""

    @staticmethod
    @patch.object(SymbolWatchFinder, '_apply_symbol_hooks_for_module')
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.importlib.import_module')
    def test_on_symbol_module_loaded_direct_match(mock_import_module, mock_apply_hooks, 
                                                 symbol_watch_finder, sample_config):
        """测试模块加载回调 - 直接匹配"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        # 执行回调
        symbol_watch_finder._on_symbol_module_loaded('module1')
        
        # 验证应用钩子被调用
        mock_apply_hooks.assert_called_once_with('module1', [
            ('symbol_0', {'symbol': 'module1:Class1.method1'})
        ])
        mock_import_module.assert_not_called()

    @staticmethod
    @patch.object(SymbolWatchFinder, '_apply_symbol_hooks_for_module')
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.importlib.import_module')
    def test_on_symbol_module_loaded_parent_match_success(mock_import_module, mock_apply_hooks,
                                                        symbol_watch_finder, sample_config):
        """测试模块加载回调 - 父包匹配且子模块导入成功"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        # 执行回调
        symbol_watch_finder._on_symbol_module_loaded('parent.child')
        
        # 验证尝试导入子模块
        mock_import_module.assert_called_once_with('parent.child.grandchild')
        mock_apply_hooks.assert_not_called()  # 当前模块没有直接匹配

    @staticmethod
    @patch.object(SymbolWatchFinder, '_apply_symbol_hooks_for_module')
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.importlib.import_module')
    def test_on_symbol_module_loaded_parent_match_failure(mock_import_module, mock_apply_hooks,
                                                         symbol_watch_finder, sample_config):
        """测试模块加载回调 - 父包匹配但子模块导入失败"""
        mock_import_module.side_effect = ImportError("Module not found")
        
        symbol_watch_finder.load_symbol_config(sample_config)
        
        symbol_watch_finder._on_symbol_module_loaded('parent.child')
        
        mock_import_module.assert_called_once_with('parent.child.grandchild')
        mock_apply_hooks.assert_not_called()

    @staticmethod
    @patch.object(SymbolWatchFinder, '_apply_symbol_hooks_for_module')
    @patch('msserviceprofiler.vllm_profiler.symbol_watcher.importlib.import_module')
    def test_on_symbol_module_loaded_mixed_matches(mock_import_module, mock_apply_hooks,
                                                  symbol_watch_finder):
        """测试模块加载回调 - 混合匹配（直接匹配和父包匹配）"""
        config_data = [
            {'symbol': 'target.module:direct_func'},
            {'symbol': 'target.module.child:child_func'}
        ]
        symbol_watch_finder.load_symbol_config(config_data)
        
        symbol_watch_finder._on_symbol_module_loaded('target.module')
        
        # 验证直接匹配的钩子被应用
        mock_apply_hooks.assert_called_once_with('target.module', [
            ('symbol_0', {'symbol': 'target.module:direct_func'})
        ])
        # 验证尝试导入子模块
        mock_import_module.assert_called_once_with('target.module.child')


class TestLoaderWrapper:
    """测试加载器包装器功能"""

    @staticmethod
    def test_loader_wrapper_creation(symbol_watch_finder, sample_config, mock_loader, mock_spec):
        """测试加载器包装器的创建和基本功能"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        with patch('importlib.machinery.PathFinder.find_spec', return_value=mock_spec):
            result = symbol_watch_finder.find_spec('module1', None)
            wrapper = result.loader
            
            # 测试包装器属性
            assert wrapper._vllm_profiler_wrapped is True
            assert wrapper._finder == symbol_watch_finder
            
            # 测试 create_module 方法
            created_module = wrapper.create_module(mock_spec)
            mock_loader.create_module.assert_called_once_with(mock_spec)
            
            # 测试 exec_module 方法（包括回调调用）
            with patch.object(symbol_watch_finder, '_on_symbol_module_loaded') as mock_callback:
                mock_module = Mock()
                wrapper.exec_module(mock_module)
                mock_loader.exec_module.assert_called_once_with(mock_module)
                mock_callback.assert_called_once_with('module1')

    @staticmethod
    def test_loader_wrapper_no_create_module(symbol_watch_finder, sample_config):
        """测试加载器没有 create_module 方法的情况"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        # 创建没有 create_module 方法的加载器
        mock_loader = Mock(spec=['exec_module'])
        mock_loader._vllm_profiler_wrapped = False
        mock_spec = Mock(loader=mock_loader)
        
        with patch('importlib.machinery.PathFinder.find_spec', return_value=mock_spec):
            result = symbol_watch_finder.find_spec('module1', None)
            wrapper = result.loader
            
            # create_module 应该返回 None
            created_module = wrapper.create_module(mock_spec)
            assert created_module is None


class TestIntegration:
    """集成测试"""
    
    @staticmethod
    def test_integration_full_workflow(symbol_watch_finder, sample_config, mock_loader, mock_spec):
        """测试完整工作流程集成测试"""
        symbol_watch_finder.load_symbol_config(sample_config)
        
        # 模拟模块导入过程
        with patch('importlib.machinery.PathFinder.find_spec', return_value=mock_spec):
            # 调用 find_spec
            result = symbol_watch_finder.find_spec('module1', None)
            
            # 验证规范被包装
            assert result.loader != mock_loader
            
            # 模拟模块加载完成
            with patch.object(symbol_watch_finder, '_on_symbol_module_loaded') as mock_callback:
                # 执行模块加载
                mock_module = Mock()
                result.loader.exec_module(mock_module)
                
                # 验证回调被调用
                mock_callback.assert_called_once_with('module1')
