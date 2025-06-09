# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.flat_quant import (
    FakeQuantizerVisitor,
    FlatQuantQuantizerConfig,
    FlatQuantQuantizerMapVisitor,
    stat_input_hook,
    stat_tensor,
    get_n_set_parameters_byname,
    get_trainable_parameters,
    convert_config,
    quantize_model
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.flat_linear import (
    FakeQuantizedLinearConfig
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType


class TestStatFunctions:
    def test_stat_tensor(self):
        act_stats = {'test_layer': {}}
        x = torch.randn(2, 10)
        
        stat_tensor(act_stats, 'test_layer', x)
        
        assert 'input_max' in act_stats['test_layer']
        assert act_stats['test_layer']['input_max'].shape == (10,)

    def test_stat_tensor_update_max(self):
        act_stats = {'test_layer': {'input_max': torch.ones(10)}}
        x = torch.randn(2, 10) * 2
        
        stat_tensor(act_stats, 'test_layer', x)
        
        assert torch.any(act_stats['test_layer']['input_max'] > 1.0)

    def test_stat_input_hook_tuple_input(self):
        act_stats = {'test_layer': {}}
        x = (torch.randn(2, 10), torch.randn(2, 5))
        
        stat_input_hook(None, x, None, 'test_layer', act_stats)
        
        assert 'input_max' in act_stats['test_layer']
        assert act_stats['test_layer']['input_max'].shape == (10,)


class TestFakeQuantizerVisitor:
    def setup_method(self):
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        )
        self.config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        self.visitor = FakeQuantizerVisitor(self.model, self.config)

    def test_initialization(self):
        assert self.visitor.model == self.model
        assert self.visitor.config == self.config
        assert len(self.visitor.quantizer_dict) == 0

    def test_register_forward_hook(self):
        linear = nn.Linear(128, 64)
        
        self.visitor.register_forward_hook(linear, 'test_linear')
        
        assert 'test_linear' in self.visitor.act_stats
        assert 'test_linear' in self.visitor.hooks

    def test_remove_forward_hook_with_prefix(self):
        linear1 = nn.Linear(128, 64)
        linear2 = nn.Linear(64, 32)
        self.visitor.register_forward_hook(linear1, 'layer1.linear')
        self.visitor.register_forward_hook(linear2, 'layer2.linear')
        
        self.visitor.remove_forward_hook(prefix="layer1")
        
        assert 'layer1.linear' in self.visitor.hooks
        assert 'layer2.linear' in self.visitor.hooks

    def test_mode_transitions(self):
        mock_quantizer = Mock()
        self.visitor.quantizer_dict['test_layer'] = mock_quantizer
        
        self.visitor.to_org_mode()
        mock_quantizer.to_org_mode.assert_called_once()
        
        self.visitor.to_calib_mode()
        mock_quantizer.to_calib_mode.assert_called_once()
        
        self.visitor.to_eval_mode(quant_weight=False)
        mock_quantizer.to_eval_mode.assert_called_once()
        mock_quantizer.fake_quant_weight.assert_not_called()


class TestFlatQuantQuantizerMapVisitor:
    def setup_method(self):
        self.model = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64)
        )
        self.config = FlatQuantQuantizerConfig(w_bits=8, a_bits=8)
        self.visitor = FlatQuantQuantizerMapVisitor(self.model, self.config)

    def test_initialization(self):
        assert self.visitor.add_diag == self.config.add_diag
        assert self.visitor.diag_alpha == self.config.diag_alpha
        assert len(self.visitor.decompose_trans_dict) == 0
        assert len(self.visitor.norm_dict) == 0

    def test_mode_transitions_with_norm_and_trans(self):
        mock_norm = Mock()
        mock_trans = Mock()
        mock_pair = Mock()
        mock_pair.__str__ = Mock(return_value="test_pair")
        
        self.visitor.norm_dict['test_norm'] = mock_norm
        self.visitor.decompose_trans_dict[mock_pair] = mock_trans
        
        # Test org mode
        self.visitor.to_org_mode()
        mock_norm.to_org_mode.assert_called_once()
        
        # Test calib mode
        with patch.object(self.visitor, '_init_diag_scale'):
            self.visitor.to_calib_mode()
        mock_norm.to_calib_mode.assert_called_once()
        
        # Test eval mode
        with patch.object(self.visitor, '_reparameterize_act_diag_scale'):
            self.visitor.to_eval_mode()
        mock_norm.to_eval_mode.assert_called_once()
        mock_trans.to_eval_mode.assert_called_once()


class TestUtilityFunctions:
    def test_get_n_set_parameters_byname(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64)
        )
        
        params = get_n_set_parameters_byname(model, ['weight'])
        
        assert len(params) == 2
        for param in params:
            assert param.requires_grad is True

    def test_convert_config_w4a4_dynamic(self):
        model = nn.Linear(128, 64)
        quant_config = Mock()
        quant_config.model_quant_type = QuantType.W4A4_DYNAMIC
        flat_quant_config = FlatQuantQuantizerConfig()
        
        config, visitor = convert_config(quant_config, flat_quant_config, model)
        
        assert config == flat_quant_config
        assert isinstance(visitor, FlatQuantQuantizerMapVisitor)

    def test_convert_config_w8a8_dynamic(self):
        model = nn.Linear(128, 64)
        quant_config = Mock()
        quant_config.model_quant_type = QuantType.W8A8_DYNAMIC
        quant_config.w_bit = 8
        quant_config.a_bit = 8
        quant_config.w_sym = True
        quant_config.a_sym = True
        quant_config.is_dynamic = True
        flat_quant_config = FlatQuantQuantizerConfig()
        
        config, visitor = convert_config(quant_config, flat_quant_config, model)
        
        assert isinstance(config, FlatQuantQuantizerConfig)
        assert isinstance(visitor, FakeQuantizerVisitor)

    def test_convert_config_float(self):
        model = nn.Linear(128, 64)
        quant_config = Mock()
        quant_config.model_quant_type = QuantType.FLOAT
        flat_quant_config = FlatQuantQuantizerConfig()
        
        config, visitor = convert_config(quant_config, flat_quant_config, model)
        
        assert config is None
        assert visitor is None

    def test_convert_config_invalid_type(self):
        model = nn.Linear(128, 64)
        quant_config = Mock()
        quant_config.model_quant_type = "INVALID_TYPE"
        flat_quant_config = FlatQuantQuantizerConfig()
        
        with pytest.raises(ValueError):
            convert_config(quant_config, flat_quant_config, model)

    def test_quantize_model(self):
        model = nn.Sequential(nn.Linear(128, 64))
        model_bridge = Mock()
        
        mock_pair = Mock()
        mock_pair.contain.return_value = True
        mock_pair.name = "test_pair"
        mock_pair.accept = Mock()
        
        pairs_dict = {
            'AttnNormLinearPair': [mock_pair],
            'AttnLinearLinearPair': [],
            'MLPNormLinearPair': [],
            'MLPLinearLinearPair': []
        }
        model_bridge.get_structure_pairs.return_value = pairs_dict
        model_bridge.model = model
        
        layer_map = {'test_layer': Mock()}
        layer_map['test_layer'].model_quant_type = QuantType.W4A4_DYNAMIC
        flat_quant_config = FlatQuantQuantizerConfig()
        
        with patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.flat_quant.QuantizerMapper') as mock_mapper_class:
            mock_mapper = Mock()
            mock_mapper_class.return_value = mock_mapper
            
            result = quantize_model(model_bridge, layer_map, flat_quant_config)
            
            mock_mapper.register_pattern.assert_called()
            mock_mapper.apply_quantizer.assert_called_once_with(pairs_dict)
            assert result == mock_mapper

    def test_quantize_model_mixed_quant_type_error(self):
        model = nn.Sequential(nn.Linear(128, 64))
        model_bridge = Mock()
        
        mock_pair = Mock()
        mock_pair.contain.return_value = True
        mock_pair.name = "test_pair"
        
        pairs_dict = {
            'AttnNormLinearPair': [mock_pair],
            'AttnLinearLinearPair': [],
            'MLPNormLinearPair': [],
            'MLPLinearLinearPair': []
        }
        model_bridge.get_structure_pairs.return_value = pairs_dict
        
        quant_config1 = Mock()
        quant_config1.model_quant_type = QuantType.W4A4_DYNAMIC
        quant_config2 = Mock()
        quant_config2.model_quant_type = QuantType.W8A8_DYNAMIC
        
        layer_map = {
            'layer1': quant_config1,
            'layer2': quant_config2
        }
        
        flat_quant_config = FlatQuantQuantizerConfig()
        
        with pytest.raises(ValueError, match="Find different quant type"):
            quantize_model(model_bridge, layer_map, flat_quant_config) 