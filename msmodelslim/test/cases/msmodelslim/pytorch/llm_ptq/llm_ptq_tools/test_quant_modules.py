# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest

import torch

from msmodelslim import logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import (
    Quantizer,
    LinearQuantizer
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    linear_quantization_params,
    init_weight_quant_normal,
    init_weight_quant_hessian
)


class TestQuantizer:
    def test_new_quant_tensor_should_return_expected_result(self):
        # 普通a_bit=8场景的激活的Quantizer.new_quant_tensor测试
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.input_scale = torch.tensor(0.5)
        test_quantizer.input_offset = torch.tensor(0.0)
        in_param_data = torch.tensor([0.4, 1, 1.6, 2])

        result = test_quantizer.new_quant_tensor(in_param_data)
        expected_result = torch.tensor([0.5, 1.0, 1.5, 2.0])
        assert torch.equal(result, expected_result)

    def test_intput_quantizer_tensor_forward_should_return_expected_result(self):
        # 普通的a_bit=8场景的激活的Quantizer.tensor_forward测试
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.init_act_and_observer(cfg)
        in_param_data = torch.tensor([1.0, 1.0, 1.0, 1.0])

        result = test_quantizer.tensor_forward(in_param_data)
        input_scale = test_quantizer.input_scale
        input_offset = test_quantizer.input_offset

        expected_result = torch.tensor([1.0, 1.0, 1.0, 1.0])
        expected_input_scale, expected_input_offset = linear_quantization_params(
            bit=8, x_min=torch.tensor([1]), x_max=torch.tensor([1]), q_signed=True, sym=False
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(input_scale, expected_input_scale)
        assert torch.equal(input_offset, expected_input_offset)

    def test_intput_quantizer_tensor_forward_with_dynamic_quant_should_return_expected_result(self):
        # 普通的a_bit=8动态量化场景的激活Quantizer.tensor_forward测试
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=True
        )
        test_quantizer.init_act_and_observer(cfg)
        in_param_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        result = test_quantizer.tensor_forward(in_param_data)
        input_scale = test_quantizer.input_scale
        input_offset = test_quantizer.input_offset

        expected_result = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        expected_input_scale, expected_input_offset = linear_quantization_params(
            bit=8, x_min=torch.tensor([1]), x_max=torch.tensor([1]), q_signed=True, sym=False
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(input_scale, expected_input_scale)
        assert torch.equal(input_offset, expected_input_offset)

    def test_intput_quantizer_tensor_forward_with_int_infer_should_return_expected_result(self):
        # 普通的a_bit=8量化场景的激活Quantizer.tensor_forward int infer测试
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.init_act_and_observer(cfg)
        test_quantizer.enable_int_infer()
        in_param_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        _ = test_quantizer.tensor_forward(in_param_data)
        test_quantizer.disable_calib()
        result = test_quantizer.tensor_forward(in_param_data)
        input_scale = test_quantizer.input_scale
        input_offset = test_quantizer.input_offset

        test_quantizer.disable_calib()

        expected_result = torch.tensor([[127.0, 127.0, 127.0, 127.0]])
        expected_input_scale, expected_input_offset = linear_quantization_params(
            bit=8, x_min=torch.tensor([1]), x_max=torch.tensor([1]), q_signed=True, sym=False
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(input_scale, expected_input_scale)
        assert torch.equal(input_offset, expected_input_offset)

    def test_weight_quantizer_tensor_forward_should_return_expected_result(self):
        # 普通的w_bit=8场景的权重的Quantizer.tensor_forward测试
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        in_param_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        result = test_quantizer.tensor_forward(in_param_data)
        weight_scale = test_quantizer.weight_scale
        weight_offset = test_quantizer.weight_offset

        expected_result = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        _, _, expected_weight_scale, expected_weight_offset = init_weight_quant_normal(
            weight=in_param_data,
            bit=8,
            is_sym=test_quantizer.is_sym,
            is_signed=test_quantizer.is_signed,
            integral_zp=True,
            admm=test_quantizer.admm,
            mm_tensor=test_quantizer.mm_per_tensor
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(weight_scale, expected_weight_scale)
        assert torch.equal(weight_offset, expected_weight_offset)

    def test_weight_quantizer_tensor_forward_with_int_infer_should_return_expected_result(self):
        # 普通的w_bit=8场景的权重的Quantizer.tensor_forward int infer测试
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        in_param_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        _ = test_quantizer.tensor_forward(in_param_data)
        test_quantizer.disable_calib()
        result = test_quantizer.tensor_forward(in_param_data)
        weight_scale = test_quantizer.weight_scale
        weight_offset = test_quantizer.weight_offset

        expected_result = torch.tensor([[127.0, 127.0, 127.0, 127.0]])
        _, _, expected_weight_scale, expected_weight_offset = init_weight_quant_normal(
            weight=in_param_data,
            bit=8,
            is_sym=test_quantizer.is_sym,
            is_signed=test_quantizer.is_signed,
            integral_zp=True,
            admm=test_quantizer.admm,
            mm_tensor=test_quantizer.mm_per_tensor
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(weight_scale, expected_weight_scale)
        assert torch.equal(weight_offset, expected_weight_offset)

    def test_init_weight_quant_normal_should_return_expected_result(self):
        # 普通的w_bit=8场景的权重的Quantizer._init_weight_quant_normal测试
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        test_quantizer._init_weight_quant_normal(in_param_weight, y=None)
        weight_scale = test_quantizer.weight_scale
        weight_offset = test_quantizer.weight_offset

        _, _, expected_weight_scale, expected_weight_offset = init_weight_quant_normal(
            weight=in_param_weight,
            bit=8,
            is_sym=test_quantizer.is_sym,
            is_signed=test_quantizer.is_signed,
            integral_zp=True,
            admm=test_quantizer.admm,
            mm_tensor=test_quantizer.mm_per_tensor
        )

        assert torch.equal(weight_scale, expected_weight_scale)
        assert torch.equal(weight_offset, expected_weight_offset)

    def test_init_weight_quant_normal_with_hessian_should_return_expected_result(self):
        # 普通的w_bit=8场景的权重的Quantizer._init_weight_quant_normal在GPTQ场景下测试
        cfg = QuantConfig(w_bit=8, a_bit=16).weight_quant(w_method="GPTQ")
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        test_quantizer._init_weight_quant_normal(in_param_input, y=in_param_input)
        weight_scale = test_quantizer.weight_scale
        weight_offset = test_quantizer.weight_offset

        _, _, expected_weight_scale, expected_weight_offset = init_weight_quant_hessian(
            weight=in_param_weight,
            input=in_param_input,
            bit=8,
            is_sym=test_quantizer.is_sym,
            is_signed=test_quantizer.is_signed,
            integral_zp=True,
            admm=test_quantizer.admm,
            mm_tensor=test_quantizer.mm_per_tensor
        )

        assert torch.equal(weight_scale, expected_weight_scale)
        assert torch.equal(weight_offset, expected_weight_offset)


class TestLinearQuantizer:
    def test_forward_should_return_expected_result(self):
        # 普通的w8a16场景的LinearQuantizer.forward测试
        cfg = QuantConfig(w_bit=8, a_bit=16).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=4, bias=False, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        linear.weight.data = in_param_weight
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)

        result = linear_quantizer(in_param_input)

        expected_result = torch.tensor([[4.]])
        assert torch.equal(result, expected_result)

    def test_forward_with_int_infer_should_return_expected_result(self):
        # 普通的w8a8场景的LinearQuantizer.forward的int_infer测试
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=1, bias=False, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                       [1.0, 1.0, 1.0, 1.0]])

        linear.weight.data = in_param_weight
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)
        linear_quantizer.quant_input.init_act_and_observer(cfg)

        _ = linear_quantizer(in_param_input)
        linear_quantizer.quant_input.disable_calib()
        linear_quantizer.quant_weight.disable_calib()
        linear_quantizer.quant_input.enable_int_infer()
        linear_quantizer.quant_weight.enable_int_infer()
        result = linear_quantizer(in_param_input)

        expected_result = torch.tensor([[4.], [4.]])
        assert torch.equal(result, expected_result)

    def test_forward_with_int_infer_and_int_bias_should_return_expected_result(self):
        # 普通的w8a8场景的带int bias的LinearQuantizer.forward的int_infer测试
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=1, bias=True, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                       [1.0, 1.0, 1.0, 1.0]])
        in_param_bias = torch.tensor([[1.]])

        linear.weight.data = in_param_weight
        linear.bias.data = in_param_bias
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)
        linear_quantizer.quant_input.init_act_and_observer(cfg)

        _ = linear_quantizer(in_param_input)
        linear_quantizer.quant_input.disable_calib()
        linear_quantizer.quant_weight.disable_calib()
        linear_quantizer.quant_input.enable_int_infer()
        linear_quantizer.quant_weight.enable_int_infer()
        linear_quantizer.quant_weight.int_bias = True
        result = linear_quantizer(in_param_input)

        expected_result = torch.tensor([[5.], [5.]])
        assert torch.equal(result, expected_result)
