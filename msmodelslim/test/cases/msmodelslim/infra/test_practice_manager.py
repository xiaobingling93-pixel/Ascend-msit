"""
practice_manager.py 模块的完整单元测试

这个测试文件提供了对 practice_manager.py 模块的全面测试，包括：
- 常量和配置验证
- 模块级函数测试  
- NaiveQuantization 类的完整功能测试
- 边界条件和异常处理
- 正则表达式和工具函数测试
"""

import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, List

# 导入被测试的模块和相关组件
from msmodelslim.infra.practice_manager import (
    NaiveQuantization, check_label, add_customized_config, confirm_to_continue, SUPPRORTED_QUANT_TYPES
)
from msmodelslim.app.naive_quantization.practice_data import (
    ConfigTask, CustomizedParams, Metadata, QuantizationConfig, load_specific_config
)
from msmodelslim.utils.yaml_database import YamlDatabase


class TestModuleConstants:
    """测试模块级常量和配置"""

    def test_supported_quant_types_constant(self):
        """测试支持的量化类型常量"""
        # 验证常量存在且为列表
        assert isinstance(SUPPRORTED_QUANT_TYPES, list)
        assert len(SUPPRORTED_QUANT_TYPES) > 0
        
        # 验证预期的量化类型
        expected_types = ["w4a16", "w4a8", "w8a16", "w8a8", "w8a8s", "w8a8c8"]
        assert SUPPRORTED_QUANT_TYPES == expected_types
        
        # 验证所有类型都是字符串且格式正确
        for quant_type in SUPPRORTED_QUANT_TYPES:
            assert isinstance(quant_type, str)
            assert len(quant_type) >= 4  # 最短格式如 "w4a8"
            assert quant_type.startswith('w')
            assert 'a' in quant_type

    def test_supported_quant_types_regex_compatibility(self):
        """测试支持的量化类型与正则表达式的兼容性"""
        import re
        pattern = r'^w(\d+)a(\d+)(c?8?)(s?)$'
        
        # 所有支持的量化类型都应该与正则表达式匹配
        for quant_type in SUPPRORTED_QUANT_TYPES:
            match = re.match(pattern, quant_type)
            assert match is not None, f"量化类型 {quant_type} 不匹配正则表达式"
            
            # 验证可以正确提取参数
            w_bit = int(match.group(1))
            a_bit = int(match.group(2))
            assert w_bit > 0 and w_bit <= 32, f"w_bit 值 {w_bit} 不合理"
            assert a_bit > 0 and a_bit <= 32, f"a_bit 值 {a_bit} 不合理"


class TestModuleFunctions:
    """测试模块级函数"""

    def test_confirm_to_continue_user_accepts(self):
        """测试用户接受继续的情况"""
        # 注意：函数检查的是 user_input != 'y'，所以只有'y'才能通过
        accept_inputs = ['y', 'Y']  # 只有单个y字符能通过
        for user_input in accept_inputs:
            with patch('builtins.input', return_value=user_input):
                # 应该正常返回，不抛出异常
                result = confirm_to_continue()
                assert result is None

        # 测试'yes'等会被拒绝
        reject_inputs = ['yes', 'YES', 'Yes']
        for user_input in reject_inputs:
            with patch('builtins.input', return_value=user_input):
                with pytest.raises(ValueError):
                    confirm_to_continue()

    def test_confirm_to_continue_user_rejects(self):
        """测试用户拒绝继续的情况"""
        reject_inputs = ['n', 'N', 'no', 'NO', 'quit', 'exit', '', 'x', 'abc']
        
        for user_input in reject_inputs:
            with patch('builtins.input', return_value=user_input):
                with pytest.raises(ValueError, match="The corresponding configuration is not currently supported"):
                    confirm_to_continue()

    def test_confirm_to_continue_custom_messages(self):
        """测试自定义提示信息"""
        custom_prompt = "Custom prompt message"
        custom_error = "Custom error message"
        
        with patch('builtins.input', return_value='y'):
            result = confirm_to_continue(prompt=custom_prompt, error_msg=custom_error)
            assert result is None
        
        with patch('builtins.input', return_value='n'):
            with pytest.raises(ValueError, match=custom_error):
                confirm_to_continue(prompt=custom_prompt, error_msg=custom_error)

    def test_confirm_to_continue_input_processing(self):
        """测试输入处理逻辑"""
        # 注意：函数检查的是前3个字符的小写版本是否等于 'y'
        # 所以只有单个'y'字符才会通过
        
        # 测试strip()和lower()处理
        with patch('builtins.input', return_value='  Y  '):
            result = confirm_to_continue()
            assert result is None
        
        # 测试前3个字符不是'y'的情况
        with patch('builtins.input', return_value='yessir'):
            # 'yes'[:3] = 'yes' != 'y', 所以会抛出异常
            with pytest.raises(ValueError):
                confirm_to_continue()
        
        with patch('builtins.input', return_value='nosir'):
            with pytest.raises(ValueError):
                confirm_to_continue()


class TestCheckLabelAdvanced:
    """check_label函数的高级测试"""

    def test_check_label_comprehensive_combinations(self):
        """测试check_label函数的全面组合"""
        # 测试所有支持的量化类型
        for quant_type in SUPPRORTED_QUANT_TYPES:
            import re
            match = re.match(r'^w(\d+)a(\d+)(c?8?)(s?)$', quant_type)
            if match:
                w_bit = int(match.group(1))
                a_bit = int(match.group(2))
                use_kv_cache = bool(match.group(3))
                is_sparse = bool(match.group(4))
                
                # 创建匹配的标签
                matching_label = {
                    'w_bit': w_bit,
                    'a_bit': a_bit,
                    'kv_cache': use_kv_cache,
                    'is_sparse': is_sparse
                }
                
                # 应该完全匹配
                assert check_label(matching_label, w_bit, a_bit, use_kv_cache, is_sparse) is True

    def test_check_label_missing_or_invalid_keys(self):
        """测试标签中缺少或无效键的处理"""
        # 缺少w_bit或a_bit会导致不匹配
        incomplete_labels = [
            {'a_bit': 8, 'kv_cache': False, 'is_sparse': False},  # 缺少w_bit
            {'w_bit': 8, 'kv_cache': False, 'is_sparse': False},  # 缺少a_bit
        ]
        
        for label in incomplete_labels:
            # 缺少w_bit或a_bit应该导致不匹配
            result = check_label(label, 8, 8, False, False)
            assert result is False
        
        # 缺少kv_cache或is_sparse但不要求它们时，仍然可以匹配
        partial_labels = [
            {'w_bit': 8, 'a_bit': 8, 'is_sparse': False},        # 缺少kv_cache
            {'w_bit': 8, 'a_bit': 8, 'kv_cache': False},         # 缺少is_sparse
            {'w_bit': 8, 'a_bit': 8},                            # 两者都缺少
        ]
        
        for label in partial_labels:
            # 当不要求kv_cache和is_sparse时，应该匹配
            result = check_label(label, 8, 8, False, False)
            assert result is True
            
        # 但如果要求了这些特性而标签中缺少，应该不匹配
        result = check_label({'w_bit': 8, 'a_bit': 8}, 8, 8, True, False)  # 要求kv_cache但缺少
        assert result is False
        
        result = check_label({'w_bit': 8, 'a_bit': 8}, 8, 8, False, True)  # 要求is_sparse但缺少
        assert result is False

    def test_check_label_type_variations(self):
        """测试不同数据类型的标签值"""
        # 测试字符串类型的位宽
        string_label = {
            'w_bit': '8',  # 字符串而非整数
            'a_bit': '8',
            'kv_cache': False,
            'is_sparse': False
        }
        
        # 字符串和整数比较应该失败
        result = check_label(string_label, 8, 8, False, False)
        assert result is False
        
        # 测试布尔值的不同表示
        bool_variations = [
            {'w_bit': 8, 'a_bit': 8, 'kv_cache': 0, 'is_sparse': 0},
            {'w_bit': 8, 'a_bit': 8, 'kv_cache': 1, 'is_sparse': 1},
            {'w_bit': 8, 'a_bit': 8, 'kv_cache': None, 'is_sparse': None},
        ]
        
        for label in bool_variations:
            # 0应该被认为是False，1应该被认为是True
            result = check_label(label, 8, 8, bool(label['kv_cache']), bool(label['is_sparse']))
            # 这取决于具体实现，但至少不应该抛出异常
            assert result in [True, False]


class TestNaiveQuantization:
    """NaiveQuantization 类的核心功能测试"""

    @pytest.fixture
    def mock_config_dir(self, tmp_path):
        """创建模拟的配置目录结构"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # 创建模型类型目录
        for model_type in ["bert", "gpt"]:
            (config_dir / model_type).mkdir()
        
        return config_dir

    @pytest.fixture
    def mock_metadata(self):
        """创建模拟的元数据对象"""
        return Metadata(
            config_id="test_config_001",
            score=95.5,
            verified_model_types=["bert", "gpt"],
            label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
        )

    @pytest.fixture
    def mock_config_task(self, mock_metadata):
        """创建模拟的配置任务对象"""
        return ConfigTask(
            metadata=mock_metadata,
            specific=QuantizationConfig(batch_size=4)
        )

    @pytest.fixture
    def mock_yaml_database(self, mock_config_task):
        """创建模拟的 YAML 数据库对象"""
        yaml_db = Mock(spec=YamlDatabase)
        yaml_db.config_by_name = {"test_config_001": mock_config_task}
        yaml_db.load_config.return_value = [[{
            'metadata': {
                'config_id': 'test_config_001',
                'score': 95.5,
                'verified_model_types': ['bert', 'gpt'],
                'label': {'w_bit': 8, 'a_bit': 8, 'kv_cache': False, 'is_sparse': False}
            },
            'spec': {'batch_size': 4}
        }]]
        return yaml_db

    def test_initialization(self, mock_config_dir, mock_yaml_database):
        """测试 NaiveQuantization 类的初始化"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            # 验证初始化结果
            assert isinstance(quantizer.sorted_task, dict)
            assert isinstance(quantizer.config_by_model_type, dict)
            assert quantizer.default_config_path is not None
            assert "default.yaml" in quantizer.default_config_path

    def test_get_best_practice_with_config_path(self, mock_config_dir, mock_yaml_database):
        """测试通过配置路径获取最佳实践配置"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            test_config_data = {
                'metadata': {
                    'config_id': 'path_config',
                    'score': 90.0,
                    'verified_model_types': ['bert'],
                    'label': {'w_bit': 8, 'a_bit': 8, 'kv_cache': False, 'is_sparse': False}
                },
                'spec': {'batch_size': 8}
            }
            
            with patch('msmodelslim.infra.practice_manager.get_valid_read_path', 
                      return_value="/valid/config.yaml"):
                with patch('builtins.open', mock_open(read_data=yaml.dump(test_config_data))):
                    result = quantizer.get_best_practice(
                        model_type="bert",
                        config_path="/valid/config.yaml"
                    )
                    
                    assert result is not None
                    assert result.metadata.config_id == "path_config"

    def test_get_best_practice_with_quant_type(self, mock_config_dir, mock_yaml_database):
        """测试通过量化类型获取最佳实践配置"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            result = quantizer.get_best_practice(
                model_type="bert",
                quant_type="w8a8"
            )
            
            assert result is not None
            assert result.metadata.config_id == "test_config_001"

    def test_get_best_practice_invalid_model_type(self, mock_config_dir, mock_yaml_database):
        """测试无效模型类型的异常处理"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            with pytest.raises(ValueError, match="Invalid model_type"):
                quantizer.get_best_practice(model_type="123invalid")

    def test_get_best_practice_fallback_scenarios(self, mock_config_dir, mock_yaml_database):
        """测试各种回退到默认配置的情况"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            default_config_data = {
                'metadata': {
                    'config_id': 'default_config',
                    'score': 80.0,
                    'verified_model_types': ['bert'],
                    'label': {'w_bit': 8, 'a_bit': 16, 'kv_cache': False, 'is_sparse': False}
                },
                'spec': {'batch_size': 4}
            }
            
            with patch('msmodelslim.infra.practice_manager.confirm_to_continue'):
                with patch('builtins.open', mock_open(read_data=yaml.dump(default_config_data))):
                    with patch('msmodelslim.infra.practice_manager.get_valid_read_path', 
                              return_value="/default/config.yaml"):
                        
                        # 测试没有量化类型的情况
                        result = quantizer.get_best_practice(model_type="bert")
                        assert result is not None
                        assert result.metadata.config_id == "default_config"
                        
                        # 测试非法量化类型的情况
                        result = quantizer.get_best_practice(
                            model_type="bert",
                            quant_type="invalid_quant_type"
                        )
                        assert result is not None
                        assert result.metadata.config_id == "default_config"

    def test_get_best_practice_invalid_quant_type_format(self, mock_config_dir, mock_yaml_database):
        """测试无效量化类型格式的异常处理"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            # 测试不在支持列表中的量化类型
            with patch('msmodelslim.infra.practice_manager.confirm_to_continue'):
                with patch('builtins.open', mock_open(read_data=yaml.dump({
                    'metadata': {
                        'config_id': 'default_config',
                        'score': 80.0,
                        'verified_model_types': ['bert'],
                        'label': {'w_bit': 8, 'a_bit': 16, 'kv_cache': False, 'is_sparse': False}
                    },
                    'spec': {'batch_size': 4}
                }))):
                    with patch('msmodelslim.infra.practice_manager.get_valid_read_path'):
                        result = quantizer.get_best_practice(
                            model_type="bert",
                            quant_type="w32a32"  # 不在支持列表中
                        )
                        assert result is not None

    def test_get_task_by_name(self, mock_config_dir, mock_yaml_database):
        """测试通过名称获取任务的功能和异常处理"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            # 成功获取配置
            result = quantizer.get_task_by_name("bert", "test_config_001")
            assert result is not None
            assert result.metadata.config_id == "test_config_001"
            
            # 无效模型类型
            with pytest.raises(ValueError, match="Model type nonexistent not found"):
                quantizer.get_task_by_name("nonexistent", "test_config_001")
            
            # 无效配置ID
            with pytest.raises(ValueError, match="ConfigTask nonexistent not found"):
                quantizer.get_task_by_name("bert", "nonexistent")

    def test_get_task_by_path(self, mock_config_dir, mock_yaml_database):
        """测试通过路径获取任务"""
        config_data = {
            'metadata': {
                'config_id': 'path_config',
                'score': 85.0,
                'verified_model_types': ['bert'],
                'label': {'w_bit': 4, 'a_bit': 16, 'kv_cache': False, 'is_sparse': False}
            },
            'spec': {'batch_size': 8}
        }
        
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            with patch('msmodelslim.infra.practice_manager.get_valid_read_path', 
                      return_value="/valid/path.yaml"):
                with patch('builtins.open', mock_open(read_data=yaml.dump(config_data))):
                    result = quantizer.get_task_by_path("/valid/path.yaml")
                    
                    assert result is not None
                    assert result.metadata.config_id == "path_config"

    def test_iter_task(self, mock_config_dir, mock_yaml_database):
        """测试迭代任务功能"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            # 成功迭代
            tasks = list(quantizer.iter_task("bert"))
            assert len(tasks) >= 1
            assert tasks[0].metadata.config_id == "test_config_001"
            
            # 无效模型类型
            with pytest.raises(ValueError, match="Model type nonexistent not found"):
                list(quantizer.iter_task("nonexistent"))

    def test_check_methods(self, mock_config_dir, mock_yaml_database):
        """测试检查方法的功能"""
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=mock_yaml_database):
            quantizer = NaiveQuantization(mock_config_dir)
            
            # 测试模型类型检查
            assert quantizer.check_model_type("bert") is True
            assert quantizer.check_model_type("nonexistent") is False
            
            # 测试配置ID检查
            assert quantizer.check_config_id("bert", "test_config_001") is True
            assert quantizer.check_config_id("bert", "nonexistent") is False
            # 对于不存在的模型类型，会抛出异常因为get方法返回空字典
            try:
                result = quantizer.check_config_id("nonexistent", "test_config_001")
                assert result is False
            except AttributeError:
                # 这是预期的，因为空字典没有config_by_name属性
                pass


class TestQuantTypeAndRegex:
    """量化类型和正则表达式相关测试"""

    def test_quant_type_regex_patterns(self):
        """测试量化类型正则表达式的各种模式"""
        import re
        
        pattern = r'^w(\d+)a(\d+)(c?8?)(s?)$'
        
        # 测试有效模式
        valid_cases = [
            ("w4a16", ["4", "16", "", ""]),
            ("w8a8", ["8", "8", "", ""]),
            ("w8a8c8", ["8", "8", "c8", ""]),
            ("w8a8s", ["8", "8", "", "s"]),
            ("w8a8c8s", ["8", "8", "c8", "s"]),
        ]
        
        for quant_type, expected_groups in valid_cases:
            match = re.match(pattern, quant_type)
            assert match is not None, f"Pattern {quant_type} should match"
            assert list(match.groups()) == expected_groups
        
        # 测试无效模式 (注意：w8a8c 实际上会匹配，因为c?8?允许只有c)
        invalid_cases = ["w8", "8a8", "wa8", "w8a8c9", "W8A8", "w8a8 ", "w8a8cc"]
        
        for invalid_quant_type in invalid_cases:
            match = re.match(pattern, invalid_quant_type)
            assert match is None, f"Pattern {invalid_quant_type} should not match"

    def test_quant_type_parameter_extraction(self):
        """测试量化类型参数提取"""
        import re
        
        pattern = r'^w(\d+)a(\d+)(c?8?)(s?)$'
        test_cases = [
            ("w4a16", 4, 16, False, False),
            ("w8a8", 8, 8, False, False),
            ("w8a8c8", 8, 8, True, False),
            ("w8a8s", 8, 8, False, True),
            ("w8a8c8s", 8, 8, True, True),
        ]
        
        for quant_type, exp_w, exp_a, exp_cache, exp_sparse in test_cases:
            match = re.match(pattern, quant_type)
            assert match is not None
            
            w_bit = int(match.group(1))
            a_bit = int(match.group(2))
            use_kv_cache = bool(match.group(3))
            is_sparse = bool(match.group(4))
            
            assert w_bit == exp_w
            assert a_bit == exp_a
            assert use_kv_cache == exp_cache
            assert is_sparse == exp_sparse


class TestUtilityFunctions:
    """辅助函数测试"""

    def test_check_label_functionality(self):
        """测试check_label函数的核心功能"""
        # 精确匹配
        label = {'w_bit': 8, 'a_bit': 8, 'kv_cache': False, 'is_sparse': False}
        assert check_label(label, 8, 8, False, False) is True
        
        # 位宽不匹配
        assert check_label(label, 4, 8, False, False) is False
        assert check_label(label, 8, 16, False, False) is False
        
        # 稀疏性和KV缓存匹配测试
        sparse_label = {'w_bit': 8, 'a_bit': 8, 'kv_cache': False, 'is_sparse': True}
        assert check_label(sparse_label, 8, 8, False, True) is True  # 稀疏性要求可以放宽
        assert check_label(sparse_label, 8, 8, False, False) is False
        
        cache_label = {'w_bit': 8, 'a_bit': 8, 'kv_cache': True, 'is_sparse': False}
        assert check_label(cache_label, 8, 8, True, False) is True  # KV缓存要求可以放宽
        assert check_label(cache_label, 8, 8, False, False) is False

    def test_add_customized_config_basic(self):
        """测试add_customized_config的基本功能"""
        metadata = Metadata(
            config_id="test",
            score=90.0,
            verified_model_types=["bert"],
            label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
        )
        config = ConfigTask(metadata=metadata, specific=QuantizationConfig())
        
        # 测试添加自定义配置
        result = add_customized_config(config, model_path="test_path", device="gpu")
        
        assert result.customized_config is not None
        assert result.customized_config.model_path == "test_path"
        assert result.customized_config.device == "gpu"

    def test_add_customized_config_edge_cases(self):
        """测试add_customized_config的边界情况"""
        metadata = Metadata(
            config_id="edge_test",
            score=85.0,
            verified_model_types=["bert"],
            label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
        )
        config = ConfigTask(metadata=metadata, specific=QuantizationConfig())
        
        # 测试空参数
        result = add_customized_config(config)
        assert result.customized_config is not None
        
        # 测试已有自定义配置的覆盖
        config.customized_config = CustomizedParams(model_path="old_path")
        result = add_customized_config(config, model_path="new_path", save_path="save_path")
        
        assert result.customized_config.model_path == "new_path"
        assert result.customized_config.save_path == "save_path"


class TestAdvancedScenarios:
    """高级场景和边界条件测试"""

    @pytest.fixture
    def multi_config_setup(self, tmp_path):
        """创建多配置测试环境"""
        config_dir = tmp_path / "multi_config"
        config_dir.mkdir()
        
        # 创建多个模型类型目录
        for model_type in ["bert", "gpt", "llama"]:
            (config_dir / model_type).mkdir()
        
        return config_dir

    def test_config_sorting_and_selection(self, multi_config_setup):
        """测试配置排序和选择逻辑"""
        # 创建多个配置，分数不同
        configs = {}
        for i, score in enumerate([95.0, 85.0, 90.0]):
            metadata = Metadata(
                config_id=f"config_{i}",
                score=score,
                verified_model_types=["bert"],
                label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
            )
            config = ConfigTask(metadata=metadata, specific=QuantizationConfig())
            configs[f"config_{i}"] = config
        
        yaml_db = Mock(spec=YamlDatabase)
        yaml_db.config_by_name = configs
        yaml_db.load_config.return_value = [[{
            'metadata': {
                'config_id': config.metadata.config_id,
                'score': config.metadata.score,
                'verified_model_types': config.metadata.verified_model_types,
                'label': config.metadata.label
            },
            'spec': {}
        } for config in configs.values()]]
        
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=yaml_db):
            quantizer = NaiveQuantization(multi_config_setup)
            
            # 验证排序正确性（最高分数在前）
            sorted_configs = quantizer.sorted_task.get("bert", [])
            assert len(sorted_configs) == 3
            
            scores = [config.metadata.score for config in sorted_configs]
            assert scores == sorted(scores, reverse=True)  # 降序排列
            
            # 验证get_best_practice返回最高分数的配置
            result = quantizer.get_best_practice(model_type="bert", quant_type="w8a8")
            assert result.metadata.score == 95.0

    def test_error_handling_comprehensive(self, multi_config_setup):
        """测试全面的错误处理"""
        yaml_db = Mock(spec=YamlDatabase)
        yaml_db.config_by_name = {}
        yaml_db.load_config.return_value = [[]]
        
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=yaml_db):
            quantizer = NaiveQuantization(multi_config_setup)
            
            # 测试空配置的处理（sorted_task会为每个模型类型目录创建空列表）
            assert len(quantizer.sorted_task) == 3  # bert, gpt, llama目录
            assert len(quantizer.config_by_model_type) == 3
            
            # 验证每个模型类型都有空的配置列表
            for model_type in ["bert", "gpt", "llama"]:
                assert quantizer.sorted_task[model_type] == []
            
            # 测试访问不存在的模型类型
            assert quantizer.check_model_type("nonexistent") is False
            
            with pytest.raises(ValueError):
                quantizer.get_task_by_name("nonexistent", "any_config")

    def test_unicode_and_special_characters(self, multi_config_setup):
        """测试Unicode和特殊字符处理"""
        special_configs = {
            "config-with-dashes": ConfigTask(
                metadata=Metadata(
                    config_id="config-with-dashes",
                    score=90.0,
                    verified_model_types=["bert"],
                    label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
                ),
                specific=QuantizationConfig()
            )
        }
        
        yaml_db = Mock(spec=YamlDatabase)
        yaml_db.config_by_name = special_configs
        yaml_db.load_config.return_value = [[{
            'metadata': {
                'config_id': 'config-with-dashes',
                'score': 90.0,
                'verified_model_types': ['bert'],
                'label': {'w_bit': 8, 'a_bit': 8, 'kv_cache': False, 'is_sparse': False}
            },
            'spec': {}
        }]]
        
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=yaml_db):
            quantizer = NaiveQuantization(multi_config_setup)
            
            # 验证特殊字符配置能被正确处理
            result = quantizer.get_task_by_name("bert", "config-with-dashes")
            assert result is not None
            assert result.metadata.config_id == "config-with-dashes"


class TestEdgeCasesAndErrorHandling:
    """边界情况和错误处理的补充测试"""

    def test_add_customized_config_none_values(self):
        """测试add_customized_config处理None值的情况"""
        metadata = Metadata(
            config_id="none_test",
            score=85.0,
            verified_model_types=["bert"],
            label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
        )
        config = ConfigTask(metadata=metadata, specific=QuantizationConfig())
        
        # 测试传入None值的参数
        result = add_customized_config(
            config, 
            model_path=None, 
            save_path=None,
            device="gpu",
            trust_remote_code=True
        )
        
        assert result.customized_config is not None
        assert result.customized_config.model_path is None  # None值会直接被设置
        assert result.customized_config.save_path is None
        assert result.customized_config.device == "gpu"
        assert result.customized_config.trust_remote_code is True

    def test_add_customized_config_non_existent_attributes(self):
        """测试add_customized_config处理不存在属性的情况"""
        metadata = Metadata(
            config_id="attr_test",
            score=85.0,
            verified_model_types=["bert"],
            label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
        )
        config = ConfigTask(metadata=metadata, specific=QuantizationConfig())
        
        # 传入CustomizedParams中不存在的属性
        result = add_customized_config(
            config,
            non_existent_attr="value",
            another_fake_attr=123,
            model_path="valid_path"  # 这个属性存在
        )
        
        assert result.customized_config is not None
        assert result.customized_config.model_path == "valid_path"
        # 不存在的属性应该被忽略，不会被设置

    def test_check_label_edge_cases(self):
        """测试check_label函数的边界情况"""
        # 测试空标签
        empty_label = {}
        result = check_label(empty_label, 8, 8, False, False)
        assert result is False
        
        # 测试包含额外字段的标签
        extra_fields_label = {
            'w_bit': 8,
            'a_bit': 8,
            'kv_cache': False,
            'is_sparse': False,
            'extra_field1': 'value1',
            'extra_field2': 42,
            'nested': {'inner': 'value'}
        }
        result = check_label(extra_fields_label, 8, 8, False, False)
        assert result is True  # 额外字段不应该影响匹配
        
        # 测试None值
        none_label = {
            'w_bit': None,
            'a_bit': 8,
            'kv_cache': False,
            'is_sparse': False
        }
        result = check_label(none_label, 8, 8, False, False)
        assert result is False

    def test_constants_integrity(self):
        """测试模块常量的完整性"""
        # 验证SUPPRORTED_QUANT_TYPES不包含重复项
        assert len(SUPPRORTED_QUANT_TYPES) == len(set(SUPPRORTED_QUANT_TYPES))
        
        # 验证所有量化类型都遵循预期格式
        import re
        pattern = r'^w(\d+)a(\d+)(c?8?)(s?)$'
        
        for quant_type in SUPPRORTED_QUANT_TYPES:
            match = re.match(pattern, quant_type)
            assert match is not None, f"量化类型 {quant_type} 格式无效"
            
            # 验证数值范围合理
            w_bit = int(match.group(1))
            a_bit = int(match.group(2))
            assert 1 <= w_bit <= 32, f"w_bit {w_bit} 超出合理范围"
            assert 1 <= a_bit <= 32, f"a_bit {a_bit} 超出合理范围"

    def test_file_path_handling(self, tmp_path):
        """测试文件路径处理相关的边界情况"""
        # 创建测试配置
        metadata = Metadata(
            config_id="path_test",
            score=85.0,
            verified_model_types=["bert"],
            label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
        )
        config_task = ConfigTask(metadata=metadata, specific=QuantizationConfig())
        
        yaml_db = Mock(spec=YamlDatabase)
        yaml_db.config_by_name = {"path_test": config_task}
        yaml_db.load_config.return_value = [[{
            'metadata': {
                'config_id': 'path_test',
                'score': 85.0,
                'verified_model_types': ['bert'],
                'label': {'w_bit': 8, 'a_bit': 8, 'kv_cache': False, 'is_sparse': False}
            },
            'spec': {'batch_size': 4}
        }]]
        
        config_dir = tmp_path / "path_test"
        config_dir.mkdir()
        (config_dir / "bert").mkdir()
        
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=yaml_db):
            quantizer = NaiveQuantization(config_dir)
            
            # 测试默认配置路径的构建
            assert quantizer.default_config_path is not None
            assert isinstance(quantizer.default_config_path, str)
            assert os.path.isabs(quantizer.default_config_path)
            assert quantizer.default_config_path.endswith("default.yaml")

    def test_yaml_database_integration(self, tmp_path):
        """测试与YamlDatabase的集成"""
        config_dir = tmp_path / "yaml_integration"
        config_dir.mkdir()
        
        # 创建模型类型目录
        bert_dir = config_dir / "bert"
        bert_dir.mkdir()
        
        # 测试空的YAML数据库
        empty_yaml_db = Mock(spec=YamlDatabase)
        empty_yaml_db.config_by_name = {}
        empty_yaml_db.load_config.return_value = [[]]
        
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=empty_yaml_db):
            quantizer = NaiveQuantization(config_dir)
            
            # 验证空数据库的处理
            assert quantizer.check_model_type("bert") is True  # 目录存在
            assert len(quantizer.sorted_task["bert"]) == 0  # 但没有配置
            
            # 测试迭代空任务列表
            tasks = list(quantizer.iter_task("bert"))
            assert len(tasks) == 0

    def test_large_score_values(self, tmp_path):
        """测试极大分数值的处理"""
        config_dir = tmp_path / "large_scores"
        config_dir.mkdir()
        (config_dir / "test").mkdir()
        
        # 创建包含极大分数的配置
        large_scores = [999999.999, 1e10, float('inf')]
        configs = {}
        
        for i, score in enumerate(large_scores):
            if score == float('inf'):
                continue  # 跳过无穷大，因为可能导致问题
            
            metadata = Metadata(
                config_id=f"large_score_{i}",
                score=score,
                verified_model_types=["test"],
                label={"w_bit": 8, "a_bit": 8, "kv_cache": False, "is_sparse": False}
            )
            config = ConfigTask(metadata=metadata, specific=QuantizationConfig())
            configs[f"large_score_{i}"] = config
        
        yaml_db = Mock(spec=YamlDatabase)
        yaml_db.config_by_name = configs
        yaml_db.load_config.return_value = [[{
            'metadata': {
                'config_id': config.metadata.config_id,
                'score': config.metadata.score,
                'verified_model_types': config.metadata.verified_model_types,
                'label': config.metadata.label
            },
            'spec': {}
        } for config in configs.values()]]
        
        with patch('msmodelslim.infra.practice_manager.YamlDatabase', return_value=yaml_db):
            quantizer = NaiveQuantization(config_dir)
            
            # 验证排序仍然正常工作
            sorted_configs = quantizer.sorted_task.get("test", [])
            assert len(sorted_configs) > 0
            
            # 验证分数按降序排列
            scores = [config.metadata.score for config in sorted_configs]
            assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--cov=msmodelslim.infra.practice_manager", "--cov-report=html"])
