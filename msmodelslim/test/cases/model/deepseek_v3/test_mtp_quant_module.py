# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from msmodelslim.model.deepseek_v3.mtp_quant_module import (
    remove_zero_and_shift,
    DeepseekV3RMSNorm,
    SharedHead,
    MTPLayer,
)


class DummyConfig(PretrainedConfig):
    """模拟配置对象"""
    model_type = "dummy"

    def __init__(self, **kwargs):
        super().__init__(
            pad_token_id=0,
            **kwargs
        )
        self.hidden_size = 128
        self.vocab_size = 1000
        self.rms_norm_eps = 1e-6
        self.num_hidden_layers = 3


class DummyDecoderLayer(nn.Module):
    """模拟DecoderLayer"""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        # 为MTP层预留属性（会被动态设置）
        self.enorm = None
        self.hnorm = None
        self.shared_head = None
        self.eh_proj = None
        self.embed_tokens = None

    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class DummyModel(nn.Module):
    """模拟基础模型的model部分"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            DummyDecoderLayer(config.hidden_size)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids=None, **kwargs):
        # 返回一个简单的输出结构
        hidden_states = torch.randn(1, 10, self.config.hidden_size)
        return type('Output', (), {'__getitem__': lambda self, i: hidden_states if i == 0 else None})()


class DummyBaseModel(nn.Module):
    """模拟完整的基础模型"""

    def __init__(self, config):
        super().__init__()
        self.model = DummyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class TestRemoveZeroAndShift(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_remove_zero_and_shift_when_matrix_has_zeros_then_shift_and_pad(self):
        """测试remove_zero_and_shift方法：矩阵包含0时应移除并前移元素"""
        matrix = torch.tensor([
            [1, 2, 0, 3, 4],
            [5, 0, 6, 7, 8],
            [0, 9, 10, 11, 12]
        ])

        result = remove_zero_and_shift(matrix)

        self.assertEqual(result.shape, matrix.shape)
        expected = torch.tensor([
            [1, 2, 3, 4, 0],
            [5, 6, 7, 8, 0],
            [9, 10, 11, 12, 0]
        ])
        self.assertTrue(torch.equal(result, expected))

    def test_remove_zero_and_shift_when_single_row_then_process_correctly(self):
        """测试remove_zero_and_shift方法：单行矩阵时应正确处理"""
        matrix = torch.tensor([[1, 0, 2, 3]])
        result = remove_zero_and_shift(matrix)
        expected = torch.tensor([[1, 2, 3, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_remove_zero_and_shift_when_called_then_preserve_device(self):
        """测试remove_zero_and_shift方法：调用时应保留设备属性"""
        matrix = torch.tensor([[1, 0, 2]], device='cpu')
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.device, matrix.device)

    def test_remove_zero_and_shift_when_called_then_preserve_dtype(self):
        """测试remove_zero_and_shift方法：调用时应保留数据类型"""
        matrix = torch.tensor([[1, 0, 2]], dtype=torch.long)
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.dtype, matrix.dtype)


class TestDeepseekV3RMSNorm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_rms_norm_initialization_when_created_then_weight_is_ones(self):
        """测试RMSNorm初始化：创建时权重应为1"""
        hidden_size = 128
        eps = 1e-6
        norm = DeepseekV3RMSNorm(hidden_size, eps=eps)

        # 验证weight参数初始化为1
        self.assertEqual(norm.weight.shape, (hidden_size,))
        self.assertTrue(torch.allclose(norm.weight, torch.ones(hidden_size)))
        self.assertEqual(norm.variance_epsilon, eps)

    def test_rms_norm_forward_when_called_then_return_normalized_output(self):
        """测试RMSNorm前向传播：调用时应返回归一化输出"""
        hidden_size = 128
        norm = DeepseekV3RMSNorm(hidden_size)

        hidden_states = torch.randn(2, 10, hidden_size)
        input_dtype = hidden_states.dtype

        # 前向传播
        output = norm(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(output.dtype, input_dtype)

    def test_rms_norm_when_different_dtypes_then_handle_correctly(self):
        """测试RMSNorm：不同数据类型时应正确处理"""
        hidden_size = 64

        # 测试float32
        norm_fp32 = DeepseekV3RMSNorm(hidden_size)
        hidden_states_fp32 = torch.randn(1, 5, hidden_size, dtype=torch.float32)
        output_fp32 = norm_fp32(hidden_states_fp32)
        self.assertEqual(output_fp32.dtype, torch.float32)

        norm_bf16 = DeepseekV3RMSNorm(hidden_size)
        hidden_states_bf16 = torch.randn(1, 5, hidden_size, dtype=torch.bfloat16)
        output_bf16 = norm_bf16(hidden_states_bf16)
        # 验证输出形状正确
        self.assertEqual(output_bf16.shape, hidden_states_bf16.shape)


class TestSharedHead(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()

    def test_shared_head_initialization_when_created_then_have_norm_and_head(self):
        """测试SharedHead初始化：创建时应包含norm和head层"""
        head = SharedHead(self.config)

        # 验证组件存在
        self.assertIsInstance(head.norm, DeepseekV3RMSNorm)
        self.assertIsInstance(head.head, nn.Linear)

        # 验证线性层配置
        self.assertEqual(head.head.in_features, self.config.hidden_size)
        self.assertEqual(head.head.out_features, self.config.vocab_size)
        self.assertIsNone(head.head.bias)

    def test_shared_head_forward_when_called_then_return_logits(self):
        """测试SharedHead前向传播：调用时应返回logits"""
        head = SharedHead(self.config)

        # 创建输入
        hidden_states = torch.randn(2, 10, self.config.hidden_size)

        # 前向传播
        logits = head(hidden_states)

        # 验证输出形状
        self.assertEqual(logits.shape, (2, 10, self.config.vocab_size))


class TestMTPLayer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = DummyConfig()

    def test_mtp_layer_initialization_when_created_then_have_all_components(self):
        """测试MTPLayer初始化：创建时应包含所有组件"""
        mtp_layer = MTPLayer(self.config)

        # 验证组件存在
        self.assertIsInstance(mtp_layer.enorm, DeepseekV3RMSNorm)
        self.assertIsInstance(mtp_layer.hnorm, DeepseekV3RMSNorm)
        self.assertIsInstance(mtp_layer.shared_head, SharedHead)
        self.assertIsInstance(mtp_layer.eh_proj, nn.Linear)
        self.assertIsInstance(mtp_layer.embed_tokens, nn.Embedding)

        # 验证线性层配置
        self.assertEqual(mtp_layer.eh_proj.in_features, self.config.hidden_size * 2)
        self.assertEqual(mtp_layer.eh_proj.out_features, self.config.hidden_size)

        # 验证嵌入层配置
        self.assertEqual(mtp_layer.embed_tokens.num_embeddings, self.config.vocab_size)
        self.assertEqual(mtp_layer.embed_tokens.embedding_dim, self.config.hidden_size)
