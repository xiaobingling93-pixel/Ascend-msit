# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
quantization.py 模块的单元测试

本测试文件为 quantization.py 中的所有类和函数编写了全面的单元测试，
包括 HookRegistry、工具函数和 Quantization 主类的测试。
使用 pytest 框架和 mock 技术来隔离外部依赖。
"""

import os
import sys
import functools
from unittest.mock import Mock, patch, MagicMock, call
import pytest

# Mock所有复杂的外部依赖
sys.modules['torch'] = Mock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch.nn.functional'] = Mock()
sys.modules['tqdm'] = Mock()
sys.modules['ascend_utils'] = Mock()
sys.modules['ascend_utils.common'] = Mock()
sys.modules['ascend_utils.common.security'] = Mock()
sys.modules['msmodelslim.tools'] = Mock()
sys.modules['msmodelslim.tools.copy_config_files'] = Mock()
sys.modules['msmodelslim.tools.logger'] = Mock()
sys.modules['msmodelslim.tools.convert_fp8_to_bf16'] = Mock()
sys.modules['msmodelslim.tools.add_safetensors'] = Mock()
sys.modules['msmodelslim.pytorch'] = Mock()
sys.modules['msmodelslim.pytorch.llm_ptq'] = Mock()
sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier'] = Mock()
sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools'] = Mock()
sys.modules['msmodelslim.utils'] = Mock()
sys.modules['msmodelslim.utils.safe_utils'] = Mock()

# 创建mock torch模块和F功能
mock_torch = Mock()
mock_torch.tensor = Mock()
mock_torch.cat = Mock()
mock_torch.float16 = "float16"
mock_torch.float32 = "float32"
mock_torch.npu = Mock()
sys.modules['torch'] = mock_torch

mock_F = Mock()
sys.modules['torch.nn.functional'] = mock_F

# 模拟数据结构
class MockMetadata:
    def __init__(self, config_id, score, verified_model_types, label):
        self.config_id = config_id
        self.score = score
        self.verified_model_types = verified_model_types
        self.label = label

class MockQuantizationConfig:
    def __init__(self, **kwargs):
        self.tokenizer_cfg = kwargs.get('tokenizer_cfg', {})
        self.model_cfg = kwargs.get('model_cfg', {})
        self.anti_cfg = kwargs.get('anti_cfg', None)
        self.anti_params = kwargs.get('anti_params', {})
        self.calib_cfg = kwargs.get('calib_cfg', {})
        self.calib_params = kwargs.get('calib_params', {})
        self.calib_save_params = kwargs.get('calib_save_params', {})
        self.batch_size = kwargs.get('batch_size', 4)
        self.anti_file = kwargs.get('anti_file', None)
        self.calib_file = kwargs.get('calib_file', None)

class MockCustomizedParams:
    def __init__(self, model_path="", save_path="", device="npu", trust_remote_code=False):
        self.model_path = model_path
        self.save_path = save_path
        self.device = device
        self.trust_remote_code = trust_remote_code

class MockConfigTask:
    def __init__(self, metadata, specific, customized_config=None):
        self.metadata = metadata
        self.specific = specific
        self.customized_config = customized_config

# 现在导入被测试的模块
from msmodelslim.app.naive_quantization.quantization import (
    HookRegistry,
    custom_hook,
    get_padding_data,
    get_batch_tokenized_data,
    get_tokenized_data,
    convert_model,
    add_safetensors_func,
    Quantization
)


class TestHookRegistry:
    """测试钩子注册器类"""

    def test_init(self):
        """测试初始化方法"""
        hook_registry = HookRegistry()
        assert hook_registry.functions == {}

    def test_register_single_function(self):
        """测试注册单个函数"""
        hook_registry = HookRegistry()
        test_func = lambda x: x * 2
        
        hook_registry.register("test_hook", "test_function", test_func)
        
        assert "test_hook" in hook_registry.functions
        assert "test_function" in hook_registry.functions["test_hook"]
        assert hook_registry.functions["test_hook"]["test_function"] is test_func

    def test_register_multiple_functions_same_hook(self):
        """测试在同一个钩子下注册多个函数"""
        hook_registry = HookRegistry()
        func1 = lambda x: x * 2
        func2 = lambda x: x + 1
        
        hook_registry.register("test_hook", "func1", func1)
        hook_registry.register("test_hook", "func2", func2)
        
        assert len(hook_registry.functions["test_hook"]) == 2
        assert hook_registry.functions["test_hook"]["func1"] is func1
        assert hook_registry.functions["test_hook"]["func2"] is func2

    def test_register_multiple_hooks(self):
        """测试注册多个不同的钩子"""
        hook_registry = HookRegistry()
        func1 = lambda x: x * 2
        func2 = lambda x: x + 1
        
        hook_registry.register("hook1", "func1", func1)
        hook_registry.register("hook2", "func2", func2)
        
        assert "hook1" in hook_registry.functions
        assert "hook2" in hook_registry.functions
        assert hook_registry.functions["hook1"]["func1"] is func1
        assert hook_registry.functions["hook2"]["func2"] is func2

    def test_get_existing_function(self):
        """测试获取已存在的函数"""
        hook_registry = HookRegistry()
        test_func = lambda x: x * 2
        hook_registry.register("test_hook", "test_function", test_func)
        
        result = hook_registry.get("test_hook", "test_function")
        assert result is test_func

    def test_get_nonexistent_hook(self):
        """测试获取不存在的钩子"""
        hook_registry = HookRegistry()
        result = hook_registry.get("nonexistent_hook", "test_function")
        assert result is None

    def test_get_nonexistent_function(self):
        """测试获取存在钩子但不存在的函数"""
        hook_registry = HookRegistry()
        hook_registry.register("test_hook", "existing_function", lambda x: x)
        
        result = hook_registry.get("test_hook", "nonexistent_function")
        assert result is None


class TestCustomHook:
    """测试自定义钩子函数"""

    def test_custom_hook_modifies_config(self):
        """测试自定义钩子正确修改模型配置"""
        model_config = {}
        custom_hook(model_config)
        
        assert model_config["mla_quantize"] == "w8a8"
        assert model_config["quantize"] == "w8a8_dynamic"
        assert model_config["moe_quantize"] == "w4a8_dynamic"
        assert model_config["model_type"] == "deepseekv2"

    def test_custom_hook_overwrites_existing_config(self):
        """测试自定义钩子覆盖已存在的配置"""
        model_config = {
            "mla_quantize": "old_value",
            "quantize": "old_value",
            "existing_key": "should_remain"
        }
        custom_hook(model_config)
        
        assert model_config["mla_quantize"] == "w8a8"
        assert model_config["quantize"] == "w8a8_dynamic"
        assert model_config["moe_quantize"] == "w4a8_dynamic"
        assert model_config["model_type"] == "deepseekv2"
        assert model_config["existing_key"] == "should_remain"


class TestGetPaddingData:
    """测试数据填充函数"""

    def test_get_padding_data_single_sequence(self):
        """测试单个序列的数据填充"""
        # 准备mock对象
        mock_tokenizer = Mock()
        mock_input_ids = Mock()
        mock_input_ids.size.return_value = 3
        mock_tokenizer.return_value.data = {'input_ids': mock_input_ids}
        
        mock_device_tensor = Mock()
        mock_device_tensor.size.return_value = 3  # 设置device tensor的size方法
        mock_input_ids.to.return_value = mock_device_tensor
        
        # 设置torch.nn.functional.pad的返回值
        with patch('msmodelslim.app.naive_quantization.quantization.F.pad') as mock_pad, \
             patch('msmodelslim.app.naive_quantization.quantization.torch.cat') as mock_cat:
            
            mock_pad.return_value = mock_device_tensor
            mock_cat.return_value = "concatenated_result"
            
            calib_list = ["test sentence"]
            device_type = "cpu"
            
            result = get_padding_data(mock_tokenizer, calib_list, device_type)
            
            # 验证tokenizer调用
            mock_tokenizer.assert_called_once_with("test sentence", return_tensors='pt', add_special_tokens=False)
            
            # 验证设备转换
            mock_input_ids.to.assert_called_once_with(device_type)
            
            # 验证填充操作
            mock_pad.assert_called_once_with(mock_device_tensor, (0, 0), value=0)
            
            # 验证拼接操作
            mock_cat.assert_called_once_with([mock_device_tensor])
            
            assert result == ["concatenated_result"]

    def test_get_padding_data_multiple_sequences(self):
        """测试多个序列的数据填充，长度不同"""
        # 准备mock对象
        mock_tokenizer = Mock()
        
        # 模拟两个不同长度的序列
        mock_input_ids_1 = Mock()
        mock_input_ids_1.size.return_value = 3
        mock_device_tensor_1 = Mock()
        mock_device_tensor_1.size.return_value = 3  # 设置device tensor的size方法
        mock_input_ids_1.to.return_value = mock_device_tensor_1
        
        mock_input_ids_2 = Mock()
        mock_input_ids_2.size.return_value = 5  # 更长的序列
        mock_device_tensor_2 = Mock()
        mock_device_tensor_2.size.return_value = 5  # 设置device tensor的size方法
        mock_input_ids_2.to.return_value = mock_device_tensor_2
        
        # 设置tokenizer返回值
        mock_tokenizer.side_effect = [
            Mock(data={'input_ids': mock_input_ids_1}),
            Mock(data={'input_ids': mock_input_ids_2})
        ]
        
        # 设置填充后的结果
        with patch('msmodelslim.app.naive_quantization.quantization.F.pad') as mock_pad, \
             patch('msmodelslim.app.naive_quantization.quantization.torch.cat') as mock_cat:
            
            padded_tensor_1 = Mock()
            padded_tensor_2 = Mock()
            mock_pad.side_effect = [padded_tensor_1, padded_tensor_2]
            mock_cat.return_value = "concatenated_result"
            
            calib_list = ["short", "longer sentence"]
            device_type = "cpu"
            
            result = get_padding_data(mock_tokenizer, calib_list, device_type)
            
            # 验证tokenizer调用次数
            assert mock_tokenizer.call_count == 2
            
            # 验证填充调用（第一个序列需要填充到最大长度5）
            expected_calls = [
                call(mock_device_tensor_1, (0, 2), value=0),  # 填充2个位置
                call(mock_device_tensor_2, (0, 0), value=0)   # 无需填充
            ]
            mock_pad.assert_has_calls(expected_calls)
            
            # 验证拼接操作
            mock_cat.assert_called_once_with([padded_tensor_1, padded_tensor_2])
            
            assert result == ["concatenated_result"]


class TestGetBatchTokenizedData:
    """测试批量tokenized数据函数"""

    @patch('msmodelslim.app.naive_quantization.quantization.get_padding_data')
    def test_get_batch_tokenized_data_single_batch(self, mock_get_padding_data):
        """测试单个批次的数据处理"""
        mock_tokenizer = Mock()
        mock_get_padding_data.return_value = ["padded_data"]
        
        calib_list = ["sentence1", "sentence2"]
        batch_size = 2
        max_len = 10
        device = "cpu"
        
        result = get_batch_tokenized_data(mock_tokenizer, calib_list, batch_size, max_len, device)
        
        # 验证get_padding_data被调用
        mock_get_padding_data.assert_called_once_with(mock_tokenizer, calib_list, device)
        
        assert result == [["padded_data"]]

    @patch('msmodelslim.app.naive_quantization.quantization.get_padding_data')
    def test_get_batch_tokenized_data_multiple_batches(self, mock_get_padding_data):
        """测试多个批次的数据处理"""
        mock_tokenizer = Mock()
        mock_get_padding_data.side_effect = [["batch1_data"], ["batch2_data"]]
        
        calib_list = ["sent1", "sent2", "sent3"]
        batch_size = 2
        max_len = 10
        device = "cpu"
        
        result = get_batch_tokenized_data(mock_tokenizer, calib_list, batch_size, max_len, device)
        
        # 验证get_padding_data被调用两次（两个批次）
        assert mock_get_padding_data.call_count == 2
        
        # 验证调用参数
        expected_calls = [
            call(mock_tokenizer, ["sent1", "sent2"], device),
            call(mock_tokenizer, ["sent3"], device)
        ]
        mock_get_padding_data.assert_has_calls(expected_calls)
        
        assert result == [["batch1_data"], ["batch2_data"]]

    @patch('msmodelslim.app.naive_quantization.quantization.get_padding_data')
    def test_get_batch_tokenized_data_long_strings_truncation(self, mock_get_padding_data):
        """测试长字符串截断功能"""
        mock_tokenizer = Mock()
        mock_get_padding_data.side_effect = [["batch1"], ["batch2"], ["batch3"]]
        
        # 创建一个长字符串，会被截断成多个部分
        long_string = "a" * 25  # 长度25，max_len=10，会被分成3部分
        calib_list = [long_string]
        batch_size = 1
        max_len = 10
        device = "cpu"
        
        result = get_batch_tokenized_data(mock_tokenizer, calib_list, batch_size, max_len, device)
        
        # 验证get_padding_data被调用3次（字符串被截断为3部分）
        assert mock_get_padding_data.call_count == 3
        
        # 验证截断后的字符串
        expected_calls = [
            call(mock_tokenizer, ["aaaaaaaaaa"], device),    # 前10个字符
            call(mock_tokenizer, ["aaaaaaaaaa"], device),    # 中间10个字符
            call(mock_tokenizer, ["aaaaa"], device)          # 最后5个字符
        ]
        mock_get_padding_data.assert_has_calls(expected_calls)
        
        assert result == [["batch1"], ["batch2"], ["batch3"]]


class TestGetTokenizedData:
    """测试tokenized数据函数"""

    def test_get_tokenized_data_default_names(self):
        """测试使用默认参数名的tokenized数据处理"""
        # 准备mock对象
        mock_tokenizer = Mock()
        mock_device = "cpu"
        
        # 模拟tokenizer返回的数据
        mock_inputs = Mock()
        mock_input_ids = Mock()
        mock_attention_mask = Mock()
        mock_inputs.data = {
            'input_ids': mock_input_ids,
            'attention_mask': mock_attention_mask
        }
        mock_tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        
        calib_list = ["test sentence 1", "test sentence 2"]
        
        result = get_tokenized_data(mock_tokenizer, calib_list, mock_device)
        
        # 验证tokenizer调用次数
        assert mock_tokenizer.call_count == 2
        
        # 验证tokenizer调用参数 (只检查前两个调用，忽略.to()调用)
        tokenizer_calls = [call for call in mock_tokenizer.mock_calls if not call[0].endswith('.to')]
        expected_calls = [
            call("test sentence 1", return_tensors='pt', padding=True),
            call("test sentence 2", return_tensors='pt', padding=True)
        ]
        assert tokenizer_calls == expected_calls
        
        # 验证设备转换
        assert mock_inputs.to.call_count == 2
        mock_inputs.to.assert_has_calls([call(mock_device), call(mock_device)])
        
        # 验证返回结果
        expected_result = [
            [mock_input_ids, mock_attention_mask],
            [mock_input_ids, mock_attention_mask]
        ]
        assert result == expected_result

    def test_get_tokenized_data_custom_names(self):
        """测试使用自定义参数名的tokenized数据处理"""
        # 准备mock对象
        mock_tokenizer = Mock()
        mock_device = "gpu"
        
        # 模拟tokenizer返回的数据
        mock_inputs = Mock()
        mock_custom_input_ids = Mock()
        mock_custom_attention_mask = Mock()
        mock_inputs.data = {
            'custom_input_ids': mock_custom_input_ids,
            'custom_attention_mask': mock_custom_attention_mask
        }
        mock_tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        
        calib_list = ["test sentence"]
        
        result = get_tokenized_data(
            mock_tokenizer, 
            calib_list, 
            mock_device,
            input_ids_name='custom_input_ids',
            attention_mask_name='custom_attention_mask'
        )
        
        # 验证tokenizer调用
        mock_tokenizer.assert_called_once_with("test sentence", return_tensors='pt', padding=True)
        
        # 验证设备转换
        mock_inputs.to.assert_called_once_with(mock_device)
        
        # 验证返回结果使用了自定义字段名
        expected_result = [[mock_custom_input_ids, mock_custom_attention_mask]]
        assert result == expected_result


class TestConvertModel:
    """测试模型转换函数"""

    @patch('msmodelslim.app.naive_quantization.quantization.auto_convert_model_fp8_to_bf16')
    @patch('msmodelslim.app.naive_quantization.quantization.OpsType')
    def test_convert_model(self, mock_ops_type, mock_auto_convert):
        """测试模型转换功能"""
        # 准备测试数据
        mock_model = Mock()
        model_path = "/path/to/model"
        mock_ops_type.AUTO = "AUTO"
        
        # 调用被测试函数
        result = convert_model(mock_model, model_path)
        
        # 验证auto_convert_model_fp8_to_bf16被正确调用
        mock_auto_convert.assert_called_once_with(mock_model, model_path, "AUTO")
        
        # 验证返回原始模型
        assert result is mock_model


class TestAddSafetensorsFunc:
    """测试添加safetensors函数"""

    @patch('msmodelslim.app.naive_quantization.quantization.add_safetensors')
    def test_add_safetensors_func(self, mock_add_safetensors):
        """测试添加safetensors功能"""
        model_path = "/path/to/model"
        save_path = "/path/to/save"
        
        # 调用被测试函数
        add_safetensors_func(model_path, save_path)
        
        # 验证add_safetensors被正确调用
        mock_add_safetensors.assert_called_once_with(
            org_paths=model_path,
            target_dir=save_path,
            safetensors_prefix="mtp_float",
            max_file_size_gb=5,
            prefix="model.layers.61."
        )


class TestQuantization:
    """测试主要的Quantization类"""

    def test_init(self):
        """测试Quantization类的初始化"""
        quantization = Quantization()
        
        # 验证hook_registry被正确初始化
        assert hasattr(quantization, 'hook_registry')
        assert isinstance(quantization.hook_registry, HookRegistry)
        
        # 验证钩子函数被正确注册
        convert_dtype_func = quantization.hook_registry.get("convert_dtype", "deepseekv2")
        assert convert_dtype_func is not None
        
        post_quant_func = quantization.hook_registry.get("post_quantization", "deepseekv2")
        assert post_quant_func is not None
        
        custom_hook_func = quantization.hook_registry.get("customized_hook_ds", "deepseekv2")
        assert custom_hook_func is not None

    def _create_test_config_task(self):
        """创建测试用的ConfigTask对象"""
        metadata = MockMetadata(
            config_id="test_config",
            score=1.0,
            verified_model_types=["deepseekv2"],
            label={"w_bit": 8, "a_bit": 8}
        )
        
        specific = MockQuantizationConfig(
            tokenizer_cfg={"use_fast": True},
            model_cfg={},  # 移除torch_dtype避免冲突
            anti_cfg=None,  # 设置为None以测试没有anti_cfg的情况
            anti_params={"param1": "value1"},
            calib_cfg={"w_bit": 8, "a_bit": 8},  # 移除dev_type避免冲突
            calib_params={"param2": "value2"},
            calib_save_params={"param3": "value3"},
            batch_size=4,
            anti_file="test_anti.jsonl",
            calib_file="test_calib.jsonl"
        )
        
        customized_config = MockCustomizedParams(
            model_path="/test/model/path",
            save_path="/test/save/path",
            device="npu",
            trust_remote_code=True
        )
        
        return MockConfigTask(
            metadata=metadata,
            specific=specific,
            customized_config=customized_config
        )

    def test_quant_process_missing_customized_config(self):
        """测试缺少自定义配置的情况"""
        # 创建没有customized_config的配置
        metadata = MockMetadata(
            config_id="test_config",
            score=1.0,
            verified_model_types=["deepseekv2"],
            label={"w_bit": 8, "a_bit": 8}
        )
        
        specific = MockQuantizationConfig()
        config_task = MockConfigTask(
            metadata=metadata,
            specific=specific,
            customized_config=None  # 设置为None
        )
        
        quantization = Quantization()
        
        # 验证会抛出ValueError
        with pytest.raises(ValueError, match="Required parameters are missing"):
            quantization.quant_process(config_task)

    @patch('msmodelslim.app.naive_quantization.quantization.tqdm')
    @patch('msmodelslim.app.naive_quantization.quantization.SafeGenerator')
    @patch('msmodelslim.app.naive_quantization.quantization.get_valid_read_path')
    @patch('msmodelslim.app.naive_quantization.quantization.set_logger_level')
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    def test_quant_process_no_anti_cfg(self, mock_abspath, mock_dirname, mock_set_logger,
                                     mock_get_valid_path, mock_safe_gen, mock_tqdm):
        """测试没有anti_cfg配置的情况"""
        # 创建没有anti_cfg的配置
        config_task = self._create_test_config_task()
        config_task.specific.anti_cfg = None  # 设置为None
        
        # 设置基本的mock
        mock_progress_bar = Mock()
        mock_tqdm.return_value = mock_progress_bar
        
        mock_dirname.return_value = "/current/dir"
        mock_abspath.side_effect = lambda x: f"/abs{x}"
        mock_get_valid_path.side_effect = lambda x, *args, **kwargs: x
        
        mock_safe_generator = Mock()
        mock_safe_gen.return_value = mock_safe_generator
        
        mock_auto_config = Mock()
        mock_auto_config.model_type = "unknown_model"  # 使用未知模型类型
        mock_auto_config.torch_dtype = "float16"
        mock_safe_generator.get_config_from_pretrained.return_value = mock_auto_config
        
        mock_tokenizer = Mock()
        # 设置tokenizer返回值的结构
        mock_inputs = Mock()
        mock_input_ids = Mock()
        mock_input_ids.size.return_value = 3  # 设置size方法返回值
        mock_device_tensor = Mock()
        mock_device_tensor.size.return_value = 3  # 设置to()返回值的size方法
        mock_input_ids.to.return_value = mock_device_tensor
        mock_inputs.data = {'input_ids': mock_input_ids}
        mock_tokenizer.return_value = mock_inputs
        mock_safe_generator.get_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_safe_generator.get_model_from_pretrained.return_value = mock_model
        
        mock_safe_generator.load_jsonl.return_value = ["test_data"]
        
        quantization = Quantization()
        
        with patch.object(quantization.hook_registry, 'get') as mock_get_hook:
            mock_get_hook.return_value = None  # 返回None表示没有注册的钩子
            
            with patch('msmodelslim.app.naive_quantization.quantization.QuantConfig') as mock_quant_config:
                mock_calib_cfg = Mock()
                mock_calib_cfg.w_bit = 8
                mock_calib_cfg.a_bit = 8
                mock_quant_config.return_value = mock_calib_cfg
                
                with patch('msmodelslim.app.naive_quantization.quantization.get_tokenized_data') as mock_get_tokenized:
                    mock_get_tokenized.return_value = [["token_data"]]
                    
                    with patch('msmodelslim.app.naive_quantization.quantization.Calibrator') as mock_calibrator:
                        mock_calib = Mock()
                        mock_calibrator.return_value = mock_calib
                        
                        # 这应该正常执行，不会因为anti_cfg为None而失败
                        quantization.quant_process(config_task)
        
        # 验证基本流程仍然执行
        mock_safe_generator.get_config_from_pretrained.assert_called_once()
        mock_safe_generator.get_model_from_pretrained.assert_called_once()


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
