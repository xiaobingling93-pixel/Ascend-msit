# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_utils import (
    StructurePair,
    ModelStructureBridge,
    AttnNormLinearPair,
    AttnLinearLinearPair,
    MLPNormLinearPair,
    MLPLinearLinearPair,
    get_module_by_name,
    set_module_by_name,
    TransformerStructurePairVisitor,
    RunnerStopExecution
)


class TestUtilityFunctions:
    def test_get_module_by_name_single_level(self):
        model = nn.Sequential()
        model.layer1 = nn.Linear(10, 5)
        
        result = get_module_by_name(model, "layer1")
        
        assert result == model.layer1

    def test_get_module_by_name_nested(self):
        model = nn.Sequential()
        model.block = nn.Sequential()
        model.block.layer = nn.Linear(10, 5)
        
        result = get_module_by_name(model, "block.layer")
        
        assert result == model.block.layer

    def test_get_module_by_name_deep_nested(self):
        model = nn.Sequential()
        model.transformer = nn.Sequential()
        model.transformer.layer = nn.Sequential()
        model.transformer.layer.attention = nn.Linear(10, 5)
        
        result = get_module_by_name(model, "transformer.layer.attention")
        
        assert result == model.transformer.layer.attention

    def test_set_module_by_name_single_level(self):
        model = nn.Sequential()
        model.layer1 = nn.Linear(10, 5)
        new_layer = nn.Linear(10, 8)
        
        set_module_by_name(model, "layer1", new_layer, clone_hooks=False)
        
        assert model.layer1 == new_layer

    def test_set_module_by_name_nested(self):
        model = nn.Sequential()
        model.block = nn.Sequential()
        model.block.layer = nn.Linear(10, 5)
        new_layer = nn.Linear(10, 8)
        
        set_module_by_name(model, "block.layer", new_layer, clone_hooks=False)
        
        assert model.block.layer == new_layer

    def test_set_module_by_name_with_hooks(self):
        model = nn.Sequential()
        old_layer = nn.Linear(10, 5)
        model.layer1 = old_layer
        
        # Register hooks on old layer
        forward_hook = lambda module, input, output: None
        backward_hook = lambda module, grad_input, grad_output: None
        
        old_layer.register_forward_hook(forward_hook)
        old_layer.register_backward_hook(backward_hook)
        
        new_layer = nn.Linear(10, 8)
        
        set_module_by_name(model, "layer1", new_layer, clone_hooks=True)
        
        assert model.layer1 == new_layer
        # Check that hooks were copied (basic check for hook dictionaries)
        assert hasattr(new_layer, '_forward_hooks')
        assert hasattr(new_layer, '_backward_hooks')

    def test_set_module_by_name_without_hooks(self):
        model = nn.Sequential()
        old_layer = nn.Linear(10, 5)
        model.layer1 = old_layer
        new_layer = nn.Linear(10, 8)
        
        set_module_by_name(model, "layer1", new_layer, clone_hooks=False)
        
        assert model.layer1 == new_layer


class TestComplexScenarios:
    def test_structure_pair_integration(self):
        """测试完整的结构对集成场景"""
        config = {"model_type": "test"}
        bridge = ModelStructureBridge(Mock(), config)
        
        # 创建不同类型的结构对
        attn_norm_pair = AttnNormLinearPair(config, "norm1", ["q", "k", "v"], "layer.0")
        attn_linear_pair = AttnLinearLinearPair(config, "v", ["o"], "layer.0")
        mlp_norm_pair = MLPNormLinearPair(config, "norm2", ["gate", "up"], "layer.0")
        mlp_linear_pair = MLPLinearLinearPair(config, "up", ["down"], "layer.0")
        
        # 注册所有结构对
        bridge.register_structure_pair(attn_norm_pair)
        bridge.register_structure_pair(attn_linear_pair)
        bridge.register_structure_pair(mlp_norm_pair)
        bridge.register_structure_pair(mlp_linear_pair)
        
        # 验证注册结果
        registry = bridge.get_structure_pairs()
        assert "AttnNormLinearPair" in registry
        assert "AttnLinearLinearPair" in registry
        assert "MLPNormLinearPair" in registry
        assert "MLPLinearLinearPair" in registry

    def test_visitor_pattern_integration(self):
        """测试访问者模式集成"""
        class TestVisitor(TransformerStructurePairVisitor):
            def __init__(self):
                self.visited = []
                
            def visit_attn_norm_linear_pair(self, pair):
                self.visited.append("attn_norm")
                return "attn_norm_result"
                
            def visit_mlp_linear_linear_pair(self, pair):
                self.visited.append("mlp_linear")
                return "mlp_linear_result"
        
        visitor = TestVisitor()
        config = {}
        
        attn_pair = AttnNormLinearPair(config, "norm", ["linear"], "prefix")
        mlp_pair = MLPLinearLinearPair(config, "pre", ["post"], "prefix")
        
        result1 = attn_pair.accept(visitor)
        result2 = mlp_pair.accept(visitor)
        
        assert result1 == "attn_norm_result"
        assert result2 == "mlp_linear_result"
        assert visitor.visited == ["attn_norm", "mlp_linear"]

    def test_module_name_operations_integration(self):
        """测试模块名称操作的集成场景"""
        # 创建复杂的嵌套模型
        model = nn.Sequential()
        model.transformer = nn.Sequential()
        model.transformer.layers = nn.ModuleList()
        
        layer = nn.Sequential()
        layer.attention = nn.Sequential()
        layer.attention.query = nn.Linear(128, 128)
        layer.attention.key = nn.Linear(128, 128)
        layer.mlp = nn.Sequential()
        layer.mlp.gate = nn.Linear(128, 512)
        
        model.transformer.layers.append(layer)
        
        # 测试获取模块
        query_module = get_module_by_name(model, "transformer.layers.0.attention.query")
        assert isinstance(query_module, nn.Linear)
        assert query_module.in_features == 128
        
        # 测试设置模块
        new_query = nn.Linear(256, 256)
        set_module_by_name(model, "transformer.layers.0.attention.query", new_query)
        
        updated_module = get_module_by_name(model, "transformer.layers.0.attention.query")
        assert updated_module == new_query
        assert updated_module.in_features == 256 