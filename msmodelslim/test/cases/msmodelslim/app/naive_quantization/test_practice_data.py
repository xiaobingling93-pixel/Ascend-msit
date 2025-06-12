# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
practice_data.py 的单元测试模块

本模块包含对 practice_data.py 中所有数据类和函数的全面单元测试，覆盖以下场景：
1. 数据类的实例化和默认值测试
2. 数据类字段赋值和类型验证
3. load_specific_config 函数的各种输入情况测试
4. 边界条件和异常情况测试
"""

import pytest
from dataclasses import fields
from typing import get_type_hints

from msmodelslim.app.naive_quantization.practice_data import (
    Metadata,
    QuantizationConfig,
    CustomizedParams,
    ConfigTask,
    load_specific_config
)


class TestMetadata:
    """Metadata 数据类的单元测试类"""

    def test_metadata_init_with_all_params(self):
        """
        测试 Metadata 类使用所有参数进行初始化
        
        验证点：
        1. 所有字段都被正确赋值
        2. 字段类型符合预期
        """
        # 准备测试数据
        config_id = "test_config_001"
        score = 95.5
        verified_model_types = ['LLaMa3.1-70B', 'Qwen2.5-72B', 'ChatGLM-6B']
        label = {'w_bit': 8, 'a_bit': 8, 'is_sparse': True, 'kv_cache': True}

        # 执行测试
        metadata = Metadata(
            config_id=config_id,
            score=score,
            verified_model_types=verified_model_types,
            label=label
        )

        # 验证结果
        assert metadata.config_id == config_id
        assert metadata.score == score
        assert metadata.verified_model_types == verified_model_types
        assert metadata.label == label
        
        # 验证字段类型
        assert isinstance(metadata.config_id, str)
        assert isinstance(metadata.score, float)
        assert isinstance(metadata.verified_model_types, list)
        assert isinstance(metadata.label, dict)

    def test_metadata_with_empty_lists_and_dicts(self):
        """
        测试 Metadata 类使用空列表和空字典初始化
        
        验证点：
        1. 支持空列表和空字典
        2. 字段值正确保存
        """
        # 准备测试数据
        config_id = "empty_config"
        score = 0.0
        verified_model_types = []
        label = {}

        # 执行测试
        metadata = Metadata(
            config_id=config_id,
            score=score,
            verified_model_types=verified_model_types,
            label=label
        )

        # 验证结果
        assert metadata.config_id == config_id
        assert metadata.score == score
        assert metadata.verified_model_types == []
        assert metadata.label == {}

    def test_metadata_with_complex_label(self):
        """
        测试 Metadata 类使用复杂的 label 字典
        
        验证点：
        1. 支持复杂嵌套的字典结构
        2. 各种数据类型都能正确保存
        """
        # 准备测试数据
        complex_label = {
            'quantization': {
                'weight': {'bit': 8, 'symmetric': True},
                'activation': {'bit': 8, 'symmetric': False}
            },
            'optimization': ['pruning', 'distillation'],
            'metrics': {
                'accuracy': 98.5,
                'latency': 12.3,
                'memory_usage': 2048
            },
            'enabled_features': [True, False, True]
        }

        # 执行测试
        metadata = Metadata(
            config_id="complex_config",
            score=88.8,
            verified_model_types=['BERT', 'GPT'],
            label=complex_label
        )

        # 验证结果
        assert metadata.label == complex_label
        assert metadata.label['quantization']['weight']['bit'] == 8
        assert metadata.label['optimization'] == ['pruning', 'distillation']
        assert metadata.label['metrics']['accuracy'] == 98.5

    def test_metadata_fields_definition(self):
        """
        测试 Metadata 类的字段定义
        
        验证点：
        1. 字段数量正确
        2. 字段名称正确
        3. 字段类型注解正确
        """
        # 获取字段信息
        metadata_fields = {field.name: field for field in fields(Metadata)}
        type_hints = get_type_hints(Metadata)

        # 验证字段数量
        assert len(metadata_fields) == 4

        # 验证字段名称
        expected_fields = {'config_id', 'score', 'verified_model_types', 'label'}
        assert set(metadata_fields.keys()) == expected_fields

        # 验证字段类型注解
        assert type_hints['config_id'] == str
        assert type_hints['score'] == float
        assert type_hints['label'] == dict


class TestQuantizationConfig:
    """QuantizationConfig 数据类的单元测试类"""

    def test_quantization_config_default_values(self):
        """
        测试 QuantizationConfig 类的默认值
        
        验证点：
        1. 所有字段都有正确的默认值
        2. 默认值类型正确
        """
        # 执行测试
        config = QuantizationConfig()

        # 验证默认值
        assert config.tokenizer_cfg is None
        assert config.model_cfg is None
        assert config.anti_cfg is None
        assert config.anti_params is None
        assert config.calib_cfg is None
        assert config.calib_params is None
        assert config.calib_save_params is None
        assert config.batch_size == 4
        assert config.anti_file is None
        assert config.calib_file is None

    def test_quantization_config_full_initialization(self):
        """
        测试 QuantizationConfig 类的完整初始化
        
        验证点：
        1. 所有字段都能正确赋值
        2. 字段值保存正确
        """
        # 准备测试数据
        tokenizer_cfg = {'vocab_size': 50000, 'model_type': 'bert'}
        model_cfg = {'hidden_size': 768, 'num_layers': 12}
        anti_cfg = {'method': 'gradual', 'steps': 100}
        anti_params = {'learning_rate': 0.001, 'epochs': 10}
        calib_cfg = {'samples': 1000, 'method': 'histogram'}
        calib_params = {'percentile': 99.9, 'bins': 256}
        calib_save_params = {'format': 'json', 'compress': True}
        batch_size = 16
        anti_file = '/path/to/anti.json'
        calib_file = '/path/to/calib.jsonl'

        # 执行测试
        config = QuantizationConfig(
            tokenizer_cfg=tokenizer_cfg,
            model_cfg=model_cfg,
            anti_cfg=anti_cfg,
            anti_params=anti_params,
            calib_cfg=calib_cfg,
            calib_params=calib_params,
            calib_save_params=calib_save_params,
            batch_size=batch_size,
            anti_file=anti_file,
            calib_file=calib_file
        )

        # 验证结果
        assert config.tokenizer_cfg == tokenizer_cfg
        assert config.model_cfg == model_cfg
        assert config.anti_cfg == anti_cfg
        assert config.anti_params == anti_params
        assert config.calib_cfg == calib_cfg
        assert config.calib_params == calib_params
        assert config.calib_save_params == calib_save_params
        assert config.batch_size == batch_size
        assert config.anti_file == anti_file
        assert config.calib_file == calib_file

    def test_quantization_config_partial_initialization(self):
        """
        测试 QuantizationConfig 类的部分初始化
        
        验证点：
        1. 部分字段可以单独设置
        2. 未设置的字段保持默认值
        """
        # 执行测试
        config = QuantizationConfig(
            model_cfg={'layers': 24},
            batch_size=8,
            calib_file='custom_calib.jsonl'
        )

        # 验证结果
        assert config.model_cfg == {'layers': 24}
        assert config.batch_size == 8
        assert config.calib_file == 'custom_calib.jsonl'
        
        # 验证未设置的字段保持默认值
        assert config.tokenizer_cfg is None
        assert config.anti_cfg is None
        assert config.anti_params is None

    def test_quantization_config_fields_definition(self):
        """
        测试 QuantizationConfig 类的字段定义
        
        验证点：
        1. 字段数量正确
        2. 字段名称正确
        3. 默认值设置正确
        """
        # 获取字段信息
        config_fields = {field.name: field for field in fields(QuantizationConfig)}

        # 验证字段数量
        assert len(config_fields) == 10

        # 验证有默认值的字段
        assert config_fields['batch_size'].default == 4
        assert config_fields['tokenizer_cfg'].default is None
        assert config_fields['model_cfg'].default is None


class TestCustomizedParams:
    """CustomizedParams 数据类的单元测试类"""

    def test_customized_params_default_values(self):
        """
        测试 CustomizedParams 类的默认值
        
        验证点：
        1. 所有字段都有正确的默认值
        2. 默认值类型和内容正确
        """
        # 执行测试
        params = CustomizedParams()

        # 验证默认值
        assert params.model_path == ""
        assert params.save_path == ""
        assert params.device == "npu"
        assert params.trust_remote_code is False

    def test_customized_params_full_initialization(self):
        """
        测试 CustomizedParams 类的完整初始化
        
        验证点：
        1. 所有字段都能正确赋值
        2. 不同类型的值都能正确保存
        """
        # 准备测试数据
        model_path = "/home/user/models/bert-base"
        save_path = "/home/user/output/quantized_model"
        device = "gpu"
        trust_remote_code = True

        # 执行测试
        params = CustomizedParams(
            model_path=model_path,
            save_path=save_path,
            device=device,
            trust_remote_code=trust_remote_code
        )

        # 验证结果
        assert params.model_path == model_path
        assert params.save_path == save_path
        assert params.device == device
        assert params.trust_remote_code == trust_remote_code

    def test_customized_params_different_devices(self):
        """
        测试 CustomizedParams 类支持不同的设备类型
        
        验证点：
        1. 支持多种设备类型字符串
        2. 设备类型正确保存
        """
        devices = ["npu", "gpu", "cpu", "cuda", "ascend"]
        
        for device in devices:
            params = CustomizedParams(device=device)
            assert params.device == device

    def test_customized_params_boolean_values(self):
        """
        测试 CustomizedParams 类的布尔值字段
        
        验证点：
        1. trust_remote_code 字段支持 True/False
        2. 布尔值正确保存
        """
        # 测试 True 值
        params_true = CustomizedParams(trust_remote_code=True)
        assert params_true.trust_remote_code is True

        # 测试 False 值
        params_false = CustomizedParams(trust_remote_code=False)
        assert params_false.trust_remote_code is False


class TestConfigTask:
    """ConfigTask 数据类的单元测试类"""

    def test_config_task_initialization(self):
        """
        测试 ConfigTask 类的初始化
        
        验证点：
        1. 必需字段正确赋值
        2. 可选字段正确处理
        """
        # 准备测试数据
        metadata = Metadata(
            config_id="task_001",
            score=92.5,
            verified_model_types=['BERT'],
            label={'w_bit': 8}
        )
        specific = QuantizationConfig(batch_size=8)
        customized_config = CustomizedParams(device="gpu")

        # 执行测试
        task = ConfigTask(
            metadata=metadata,
            specific=specific,
            customized_config=customized_config
        )

        # 验证结果
        assert task.metadata == metadata
        assert task.specific == specific
        assert task.customized_config == customized_config

    def test_config_task_without_customized_config(self):
        """
        测试 ConfigTask 类不设置 customized_config 的情况
        
        验证点：
        1. customized_config 默认为 None
        2. 其他字段正常工作
        """
        # 准备测试数据
        metadata = Metadata(
            config_id="task_002",
            score=85.0,
            verified_model_types=['GPT'],
            label={'a_bit': 16}
        )
        specific = QuantizationConfig(batch_size=4)

        # 执行测试
        task = ConfigTask(metadata=metadata, specific=specific)

        # 验证结果
        assert task.metadata == metadata
        assert task.specific == specific
        assert task.customized_config is None

    def test_config_task_with_complex_data(self):
        """
        测试 ConfigTask 类使用复杂数据结构
        
        验证点：
        1. 支持复杂的嵌套数据结构
        2. 所有组件正确组合
        """
        # 准备复杂测试数据
        metadata = Metadata(
            config_id="complex_task",
            score=97.8,
            verified_model_types=['LLaMa3.1-70B', 'Qwen2.5-72B'],
            label={
                'quantization': {'w_bit': 8, 'a_bit': 8},
                'optimization': ['pruning', 'distillation'],
                'performance': {'accuracy': 95.5, 'speed_up': 2.1}
            }
        )
        
        specific = QuantizationConfig(
            tokenizer_cfg={'vocab_size': 32000},
            model_cfg={'hidden_size': 4096, 'num_layers': 32},
            batch_size=16,
            calib_file='large_dataset.jsonl'
        )
        
        customized_config = CustomizedParams(
            model_path="/data/models/llama",
            save_path="/data/output/quantized",
            device="npu",
            trust_remote_code=True
        )

        # 执行测试
        task = ConfigTask(
            metadata=metadata,
            specific=specific,
            customized_config=customized_config
        )

        # 验证结果
        assert task.metadata.config_id == "complex_task"
        assert task.metadata.score == 97.8
        assert task.specific.batch_size == 16
        assert task.customized_config.device == "npu"
        
        # 验证嵌套数据结构
        assert task.metadata.label['quantization']['w_bit'] == 8
        assert task.specific.model_cfg['hidden_size'] == 4096


class TestLoadSpecificConfig:
    """load_specific_config 函数的单元测试类"""

    def test_load_specific_config_empty_dict(self):
        """
        测试 load_specific_config 函数使用空字典
        
        验证点：
        1. 空字典输入不会报错
        2. 所有字段都使用默认值
        3. 返回正确的 QuantizationConfig 实例
        """
        # 执行测试
        config = load_specific_config({})

        # 验证结果
        assert isinstance(config, QuantizationConfig)
        assert config.tokenizer_cfg == {}
        assert config.model_cfg == {}
        assert config.anti_cfg is None
        assert config.anti_params == {}
        assert config.calib_cfg == {}
        assert config.calib_params == {}
        assert config.calib_save_params == {}
        assert config.batch_size == 4
        assert config.anti_file is None
        assert config.calib_file == '../../../example/common/teacher_qualification.jsonl'

    def test_load_specific_config_full_spec(self):
        """
        测试 load_specific_config 函数使用完整配置
        
        验证点：
        1. 所有配置字段都被正确加载
        2. 字段值与输入一致
        3. 类型正确保存
        """
        # 准备测试数据
        yaml_spec = {
            'tokenizer_cfg': {'vocab_size': 50000, 'model_type': 'bert'},
            'model_cfg': {'hidden_size': 768, 'num_layers': 12},
            'anti_cfg': {'method': 'gradual', 'steps': 100},
            'anti_params': {'learning_rate': 0.001, 'epochs': 10},
            'calib_cfg': {'samples': 1000, 'method': 'histogram'},
            'calib_params': {'percentile': 99.9, 'bins': 256},
            'calib_save_params': {'format': 'json', 'compress': True},
            'batch_size': 16,
            'anti_file': '/path/to/anti.json',
            'calib_file': '/path/to/custom_calib.jsonl'
        }

        # 执行测试
        config = load_specific_config(yaml_spec)

        # 验证结果
        assert config.tokenizer_cfg == yaml_spec['tokenizer_cfg']
        assert config.model_cfg == yaml_spec['model_cfg']
        assert config.anti_cfg == yaml_spec['anti_cfg']
        assert config.anti_params == yaml_spec['anti_params']
        assert config.calib_cfg == yaml_spec['calib_cfg']
        assert config.calib_params == yaml_spec['calib_params']
        assert config.calib_save_params == yaml_spec['calib_save_params']
        assert config.batch_size == yaml_spec['batch_size']
        assert config.anti_file == yaml_spec['anti_file']
        assert config.calib_file == yaml_spec['calib_file']

    def test_load_specific_config_partial_spec(self):
        """
        测试 load_specific_config 函数使用部分配置
        
        验证点：
        1. 部分配置正确加载
        2. 缺失配置使用默认值
        3. 函数处理健壮
        """
        # 准备测试数据
        yaml_spec = {
            'model_cfg': {'hidden_size': 1024},
            'batch_size': 8,
            'calib_file': 'custom_data.jsonl'
        }

        # 执行测试
        config = load_specific_config(yaml_spec)

        # 验证设置的值
        assert config.model_cfg == {'hidden_size': 1024}
        assert config.batch_size == 8
        assert config.calib_file == 'custom_data.jsonl'

        # 验证默认值
        assert config.tokenizer_cfg == {}
        assert config.anti_cfg is None
        assert config.anti_params == {}

    def test_load_specific_config_with_none_values(self):
        """
        测试 load_specific_config 函数处理 None 值
        
        验证点：
        1. None 值被正确处理（键存在但值为None时，保持None值）
        2. 函数行为符合 dict.get() 的语义
        """
        # 准备测试数据
        yaml_spec = {
            'tokenizer_cfg': None,
            'anti_cfg': None,
            'batch_size': None,
            'calib_file': None
        }

        # 执行测试
        config = load_specific_config(yaml_spec)

        # 验证结果 (键存在但值为None时，保持None值)
        assert config.tokenizer_cfg is None  # 键存在但值为None，保持None
        assert config.anti_cfg is None
        assert config.batch_size is None  # 键存在但值为None，保持None
        assert config.calib_file is None  # 键存在但值为None，保持None

    def test_load_specific_config_with_nested_dicts(self):
        """
        测试 load_specific_config 函数处理嵌套字典
        
        验证点：
        1. 复杂的嵌套字典结构正确保存
        2. 嵌套内容可以正确访问
        """
        # 准备测试数据
        yaml_spec = {
            'model_cfg': {
                'architecture': {
                    'encoder': {'layers': 12, 'hidden_size': 768},
                    'decoder': {'layers': 6, 'hidden_size': 512}
                },
                'training': {
                    'optimizer': 'adamw',
                    'lr_schedule': {'type': 'cosine', 'warmup_steps': 1000}
                }
            },
            'calib_params': {
                'quantization': {
                    'weight': {'bits': 8, 'symmetric': True},
                    'activation': {'bits': 8, 'symmetric': False}
                }
            }
        }

        # 执行测试
        config = load_specific_config(yaml_spec)

        # 验证嵌套结构
        assert config.model_cfg['architecture']['encoder']['layers'] == 12
        assert config.model_cfg['training']['optimizer'] == 'adamw'
        assert config.calib_params['quantization']['weight']['bits'] == 8

    def test_load_specific_config_with_different_batch_sizes(self):
        """
        测试 load_specific_config 函数处理不同的 batch_size 值
        
        验证点：
        1. 不同的 batch_size 值都能正确设置
        2. 数字类型正确保存
        """
        batch_sizes = [1, 4, 8, 16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            yaml_spec = {'batch_size': batch_size}
            config = load_specific_config(yaml_spec)
            assert config.batch_size == batch_size
            assert isinstance(config.batch_size, int)

    def test_load_specific_config_return_type(self):
        """
        测试 load_specific_config 函数的返回类型
        
        验证点：
        1. 返回值类型为 QuantizationConfig
        2. 返回值具有所有必要的属性
        """
        # 执行测试
        config = load_specific_config({'model_cfg': {'test': True}})

        # 验证返回类型
        assert isinstance(config, QuantizationConfig)
        
        # 验证所有属性都存在
        expected_attrs = [
            'tokenizer_cfg', 'model_cfg', 'anti_cfg', 'anti_params',
            'calib_cfg', 'calib_params', 'calib_save_params',
            'batch_size', 'anti_file', 'calib_file'
        ]
        
        for attr in expected_attrs:
            assert hasattr(config, attr)


# 集成测试类
class TestPracticeDataIntegration:
    """practice_data.py 模块的集成测试类"""

    def test_complete_workflow(self):
        """
        测试完整的数据处理工作流程
        
        验证点：
        1. 各个组件能够正确协同工作
        2. 数据在组件间正确传递
        """
        # 步骤1：加载配置
        yaml_spec = {
            'model_cfg': {'hidden_size': 768, 'num_layers': 12},
            'batch_size': 16,
            'calib_file': 'integration_test.jsonl'
        }
        specific_config = load_specific_config(yaml_spec)

        # 步骤2：创建元数据
        metadata = Metadata(
            config_id="integration_test",
            score=95.0,
            verified_model_types=['BERT', 'RoBERTa'],
            label={'w_bit': 8, 'a_bit': 8}
        )

        # 步骤3：创建自定义参数
        customized_params = CustomizedParams(
            model_path="/models/bert",
            save_path="/output/quantized",
            device="npu",
            trust_remote_code=False
        )

        # 步骤4：创建完整任务配置
        task = ConfigTask(
            metadata=metadata,
            specific=specific_config,
            customized_config=customized_params
        )

        # 验证完整工作流程
        assert task.metadata.config_id == "integration_test"
        assert task.specific.model_cfg['hidden_size'] == 768
        assert task.customized_config.device == "npu"
        
        # 验证数据一致性
        assert task.specific.batch_size == 16
        assert task.metadata.score == 95.0

    def test_data_serialization_compatibility(self):
        """
        测试数据结构的序列化兼容性
        
        验证点：
        1. 数据类可以转换为字典
        2. 字典包含所有必要字段
        """
        # 创建测试数据
        config = QuantizationConfig(
            model_cfg={'test': True},
            batch_size=8
        )
        
        params = CustomizedParams(
            model_path="/test",
            device="gpu"
        )

        # 测试数据类转字典（通过 __dict__ 或 dataclasses.asdict）
        from dataclasses import asdict
        
        config_dict = asdict(config)
        params_dict = asdict(params)

        # 验证字典结构
        assert 'model_cfg' in config_dict
        assert 'batch_size' in config_dict
        assert 'model_path' in params_dict
        assert 'device' in params_dict
        
        # 验证值的正确性
        assert config_dict['model_cfg'] == {'test': True}
        assert config_dict['batch_size'] == 8
        assert params_dict['device'] == "gpu"


if __name__ == '__main__':
    """
    运行测试的入口点
    
    使用方法：
    1. 直接运行此文件：python test_practice_data.py
    2. 使用 pytest：pytest test_practice_data.py
    3. 带覆盖率报告：pytest --cov=msmodelslim.app.naive_quantization.practice_data test_practice_data.py
    """
    pytest.main([__file__, '-v', '--tb=short'])
