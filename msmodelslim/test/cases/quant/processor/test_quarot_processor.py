# -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from unittest.mock import MagicMock

import pytest
import torch

from resources.fake.qwen3_dense import FakeQwen3Creator

from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.quarot.hadamard import random_hadamard_matrix, walsh_matrix
from msmodelslim.quant.processor.quarot.quarot import QuaRotProcessorConfig, QuaRotProcessor
from msmodelslim.quant.processor.quarot.quarot_interface import QuaRotAdapter


@pytest.fixture
def mock_model():
    """创建模拟模型"""
    torch.manual_seed(42)
    model = FakeQwen3Creator.get_model()
    return model


@pytest.fixture
def basic_config():
    """基础配置"""
    config = QuaRotProcessorConfig()
    config.online = False
    config.block_size = -1
    config.down_proj_online_layers = []
    config.max_tp_size = 4
    return config


@pytest.fixture
def mock_adapter(mock_model):
    """创建模拟适配器"""
    return MockQuaRotAdapter(mock_model)


class TestRotMaker:
    @pytest.mark.parametrize("size", [12, 20, 28, 36, 40, 52, 60, 76, 108, 136, 140, 156, 160, 172, 200])
    def test_create_rot_with_random(self, size):
        rot = random_hadamard_matrix(size, torch.float32, torch.device("cpu"))
        assert rot.shape == (size, size)

        # 验证正交性
        identity = torch.eye(size, dtype=torch.float32)
        product = rot @ rot.T
        assert torch.allclose(product, identity, atol=1e-5)

    @pytest.mark.parametrize("size", [11, 21, 87, 121])
    def test_create_rot_with_random_unsupported(self, size):
        """
        测试random_hadamard_matrix在不支持的size下会抛出UnsupportedError异常
        """
        with pytest.raises(UnsupportedError):
            random_hadamard_matrix(size, torch.float32, torch.device("cpu"))

    @pytest.mark.parametrize("size", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    def test_create_rot_with_walsh(self, size):
        
        rot = walsh_matrix(size, torch.float32, torch.device("cpu"))
        assert rot.shape == (size, size)


class MockQuaRotAdapter(QuaRotAdapter):
    def __init__(self, model=None):
        if model is None:
            torch.manual_seed(42)
            self.model = FakeQwen3Creator.get_model()
        else:
            self.model = model

    def get_hidden_dim(self):
        return self.model.config.hidden_size

    def get_head_dim(self):
        return self.model.config.head_dim

    def get_num_attention_heads(self):
        return self.model.config.num_attention_heads

    def get_num_key_value_heads(self):
        return self.model.config.num_key_value_heads

    def get_lm_head(self):
        return "lm_head"

    def get_pre_head_layernorm(self):
        return "norm"

    def get_embedding(self):
        return "embed_tokens"

    def get_layer_wise_norm_liner_pair(self, decoder_module):
        norm_linear_pairs = {}
        norm_linear_pairs[decoder_module.input_layernorm] = [
            decoder_module.self_attn.q_proj, 
            decoder_module.self_attn.k_proj, 
            decoder_module.self_attn.v_proj
        ]
        norm_linear_pairs[decoder_module.post_attention_layernorm] = [
            decoder_module.mlp.gate_proj, 
            decoder_module.mlp.up_proj
        ]
        return norm_linear_pairs
    
    def get_layer_wise_ov_pair(self, decoder_module):
        ov_pairs = {}
        ov_pairs[decoder_module.self_attn.o_proj] = decoder_module.self_attn.v_proj
        return ov_pairs
    
    def get_layer_wise_up_down_pair(self, decoder_module):
        up_down_pairs = {}
        up_down_pairs[decoder_module.mlp.up_proj] = decoder_module.mlp.down_proj
        return up_down_pairs


class TestQuaRotAdapter:
    """测试QuaRotAdapter的所有方法"""
    @staticmethod
    def test_abstract_class():
        decoder_module = MagicMock()
        adapter = QuaRotAdapter()
        adapter.get_hidden_dim()
        adapter.get_head_dim()
        adapter.get_num_attention_heads()
        adapter.get_num_key_value_heads()
        adapter.get_lm_head()
        adapter.get_pre_head_layernorm()
        adapter.get_embedding()
        adapter.get_layer_wise_norm_liner_pair(decoder_module)
        adapter.get_layer_wise_ov_pair(decoder_module)
        adapter.get_layer_wise_up_down_pair(decoder_module)


class TestQuaRotProcessor:
    """测试QuaRotProcessor类"""

    @staticmethod
    def test_init_with_default_config(mock_model, basic_config, mock_adapter):
        """测试使用默认配置初始化"""
        processor = QuaRotProcessor(mock_model, basic_config, mock_adapter)
        assert processor.config == basic_config
        assert processor.model == mock_model
        assert processor.adapter == mock_adapter
        assert processor.rot is None
        assert processor.rot_att_v is None

    @staticmethod
    def test_pre_run_online_false(mock_model, basic_config, mock_adapter):
        """测试online=False时的pre_run流程"""
        basic_config.online = False
        processor = QuaRotProcessor(mock_model, basic_config, mock_adapter)
        processor.pre_run()
        
        # online为False时，online相关矩阵应为None
        assert processor.rot_online_down_proj is None
        assert processor.rot_online_o_proj is None

    @staticmethod
    def test_pre_run_online_true(mock_model, basic_config, mock_adapter):
        """测试online=True时的pre_run流程"""
        basic_config.online = True
        processor = QuaRotProcessor(mock_model, basic_config, mock_adapter)
        processor.pre_run()
        
        # online为True时，应创建online相关矩阵
        assert processor.rot_online_down_proj is not None
        assert processor.rot_online_o_proj is not None

    @staticmethod
    @pytest.mark.parametrize("online_flag", [True, False])
    def test_pre_run_linear_fusion(mock_model, basic_config, mock_adapter, online_flag):
        """测试pre_run阶段权重融合的有效性"""
        basic_config.online = online_flag
        embed_tokens_weight_before = mock_adapter.model.embed_tokens.weight.data.clone()

        processor = QuaRotProcessor(mock_model, basic_config, mock_adapter)
        processor.pre_run()
        
        # 检查旋转矩阵是否创建
        assert processor.rot is not None
        assert processor.rot_att_v is not None
        
        # 检查旋转矩阵维度
        model_dim = mock_adapter.get_hidden_dim()
        head_dim = mock_adapter.get_head_dim()
        assert processor.rot.shape == (model_dim, model_dim)
        assert processor.rot_att_v.shape == (head_dim, head_dim)
        
        # 验证旋转矩阵正交性
        model_dim = mock_adapter.get_hidden_dim()
        head_dim = mock_adapter.get_head_dim()
        device = mock_model.device
        
        identity_model = torch.eye(model_dim, device=device)
        identity_head = torch.eye(head_dim, device=device)
        
        assert torch.allclose(
            torch.matmul(processor.rot, processor.rot.T), 
            identity_model, 
            atol=1e-5
        )
        assert torch.allclose(
            torch.matmul(processor.rot_att_v, processor.rot_att_v.T), 
            identity_head, 
            atol=1e-5
        )

        # 验证层融合与旋转的有效性
        norm_weight_after = mock_adapter.model.norm.weight.data.clone()
        embed_tokens_weight_after = mock_adapter.model.embed_tokens.weight.data.clone()
        assert torch.allclose(norm_weight_after, torch.ones_like(norm_weight_after), atol=1e-5)
        # 检查嵌入层权重变化
        weight_diff = torch.abs(embed_tokens_weight_after - embed_tokens_weight_before)
        threshold_check = weight_diff > 1e-5
        change_ratio = torch.sum(threshold_check) / embed_tokens_weight_before.numel()
        assert change_ratio > 0.9


    @staticmethod
    @pytest.mark.parametrize("online_flag", [True, False])
    def test_preprocess_linear_fusion_online(mock_model, basic_config, mock_adapter, online_flag):
        """测试preprocess阶段层权重融合的有效性"""
        basic_config.online = online_flag
        processor = QuaRotProcessor(mock_model, basic_config, mock_adapter)
        processor.pre_run()
        
        q_proj_weight_before = mock_adapter.model.layers[0].self_attn.q_proj.weight.data.clone()
        o_proj_weight_before = mock_adapter.model.layers[0].self_attn.o_proj.weight.data.clone()
        up_proj_weight_before = mock_adapter.model.layers[0].mlp.up_proj.weight.data.clone()
        down_proj_weight_before = mock_adapter.model.layers[0].mlp.down_proj.weight.data.clone()

        batch_request = BatchProcessRequest("model.layers.0", mock_adapter.model.layers[0])
        processor.preprocess(batch_request)

        q_proj_weight_after = mock_adapter.model.layers[0].self_attn.q_proj.weight.data.clone()
        o_proj_weight_after = mock_adapter.model.layers[0].self_attn.o_proj.weight.data.clone()
        up_proj_weight_after = mock_adapter.model.layers[0].mlp.up_proj.weight.data.clone()
        down_proj_weight_after = mock_adapter.model.layers[0].mlp.down_proj.weight.data.clone()
        input_layernorm_weight_after = mock_adapter.model.layers[0].input_layernorm.weight.data.clone()
        post_attention_layernorm_weight_after = \
            mock_adapter.model.layers[0].post_attention_layernorm.weight.data.clone()
        
        # 验证层融合与旋转的有效性
        assert torch.allclose(
            input_layernorm_weight_after, 
            torch.ones_like(input_layernorm_weight_after), 
            atol=1e-5
        )
        assert torch.allclose(
            post_attention_layernorm_weight_after, 
            torch.ones_like(post_attention_layernorm_weight_after), 
            atol=1e-5
        )
        
        # 检查q_proj权重变化
        q_weight_diff = torch.abs(q_proj_weight_before - q_proj_weight_after)
        q_threshold_check = q_weight_diff > 1e-5
        q_change_ratio = torch.sum(q_threshold_check) / q_proj_weight_before.numel()
        assert q_change_ratio > 0.9
        
        # 检查o_proj权重变化
        o_weight_diff = torch.abs(o_proj_weight_before - o_proj_weight_after)
        o_threshold_check = o_weight_diff > 1e-5
        o_change_ratio = torch.sum(o_threshold_check) / o_proj_weight_before.numel()
        assert o_change_ratio > 0.9
        
        # 检查up_proj权重变化
        up_weight_diff = torch.abs(up_proj_weight_before - up_proj_weight_after)
        up_threshold_check = up_weight_diff > 1e-5
        up_change_ratio = torch.sum(up_threshold_check) / up_proj_weight_before.numel()
        assert up_change_ratio > 0.9
        
        # 检查down_proj权重变化
        down_weight_diff = torch.abs(down_proj_weight_before - down_proj_weight_after)
        down_threshold_check = down_weight_diff > 1e-5
        down_change_ratio = torch.sum(down_threshold_check) / down_proj_weight_before.numel()
        assert down_change_ratio > 0.9

    @staticmethod
    @pytest.mark.parametrize("online_flag", [True, False])
    def test_quarot_pipeline(mock_model, basic_config, mock_adapter, online_flag):
        """测试QuaRotProcessor的完整流程前后一致性"""
    
        batch_size, seq_length = 2, 8
        vocab_size = 1000
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        mock_model.eval()
        with torch.no_grad():
            output_logits_before_anti = mock_model(input_ids)

        basic_config.online = online_flag
        processor = QuaRotProcessor(mock_model, basic_config, mock_adapter)
        processor.pre_run()

        num_layers = mock_adapter.model.config.num_hidden_layers
        for i in range(num_layers):
            batch_request = BatchProcessRequest(f"model.layers.{i}", mock_adapter.model.layers[i])
            processor.preprocess(batch_request)
        
        with torch.no_grad():
            output_logits_after_anti = mock_model(input_ids)

        diff = torch.subtract(output_logits_after_anti, output_logits_before_anti)
        squared_diff = torch.pow(diff, 2)
        mean_squared_diff = torch.mean(squared_diff)
        dist = torch.sqrt(mean_squared_diff)
        assert dist.item() < 1, f"Distance: {dist.item()}"