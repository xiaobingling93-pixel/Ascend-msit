from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckLayerNormOperation, LayerNormType, QuantType, DynamicQuantType
from msit_llm.common.log import logger
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckLayerNormOperation.__bases__ = (MockOperationTest,)


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger():
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug


@pytest.mark.parametrize("layer_type, in_tensors, cur_param, expected_result", [
    (LayerNormType.LAYER_NORM_NROM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4)], {'epsilon': 1e-5}, [torch.randn(2, 4)]),
    (LayerNormType.LAYER_NORM_PRENORM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4), torch.randn(4)], {'epsilon': 1e-5, 'zoomScaleValue': 1.0}, [torch.randn(2, 4), torch.randn(2, 4)]),
    (LayerNormType.LAYER_NORM_POSTNORM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4), torch.randn(4)], {'epsilon': 1e-5, 'zoomScale': 1.0}, [torch.randn(2, 4)]),
])
def test_golden_calc_when_valid_input(layer_type, in_tensors, cur_param, expected_result):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}
    if layer_type == LayerNormType.LAYER_NORM_NROM.value:
        op.op_param['normParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_PRENORM.value:
        op.op_param['preNormParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
        op.op_param['postNormParam'] = cur_param

    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("layer_type, in_tensors, cur_param, expected_error", [
    (LayerNormType.LAYER_NORM_NROM.value, [torch.randn(2, 4), torch.randn(4)], {'epsilon': 1e-5}, RuntimeError),
    (LayerNormType.LAYER_NORM_PRENORM.value, [torch.randn(2, 4), torch.randn(4)],
     {'epsilon': 1e-5, 'zoomScaleValue': 1.0}, RuntimeError),
    (LayerNormType.LAYER_NORM_POSTNORM.value, [torch.randn(2, 4), torch.randn(4)], {'epsilon': 1e-5, 'zoomScale': 1.0},
     RuntimeError),
])
def test_golden_calc_when_invalid_input(layer_type, in_tensors, cur_param, expected_error):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}
    if layer_type == LayerNormType.LAYER_NORM_NROM.value:
        op.op_param['normParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_PRENORM.value:
        op.op_param['preNormParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
        op.op_param['postNormParam'] = cur_param

    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


@pytest.mark.parametrize("layer_type, validate_param_return, expected_execute_call", [
    (LayerNormType.LAYER_NORM_NROM.value, True, True),
    (LayerNormType.LAYER_NORM_PRENORM.value, True, True),
    (LayerNormType.LAYER_NORM_POSTNORM.value, True, True),
    (LayerNormType.LAYER_NORM_UNDEFINED.value, False, False),
])
def test_test_when_valid_input(layer_type, validate_param_return, expected_execute_call):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}

    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()


@pytest.mark.parametrize("layer_type, in_tensors, cur_param, expected_result", [
    (LayerNormType.LAYER_NORM_NROM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_TENSOR.value,
      'dynamicQuantType': DynamicQuantType.DYNAMIC_QUANT_SYMMETRIC.value}, [torch.randn(2, 4), torch.randn(2, 4)]),
    (LayerNormType.LAYER_NORM_POSTNORM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_TENSOR.value,
      'dynamicQuantType': DynamicQuantType.DYNAMIC_QUANT_SYMMETRIC.value}, [torch.randn(2, 4), torch.randn(2, 4)]),
])
def test_golden_calc_when_dynamic_quant_symmetric(layer_type, in_tensors, cur_param, expected_result):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}
    if layer_type == LayerNormType.LAYER_NORM_NROM.value:
        op.op_param['normParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
        op.op_param['postNormParam'] = cur_param

    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("layer_type, in_tensors, cur_param, expected_result", [
    (LayerNormType.LAYER_NORM_NROM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_TENSOR.value,
      'dynamicQuantType': DynamicQuantType.DYNAMIC_QUANT_ASYMMETRIC.value}, [torch.randn(2, 4), torch.randn(2, 4)]),
    (LayerNormType.LAYER_NORM_POSTNORM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_TENSOR.value,
      'dynamicQuantType': DynamicQuantType.DYNAMIC_QUANT_ASYMMETRIC.value}, [torch.randn(2, 4), torch.randn(2, 4)]),
])
def test_golden_calc_when_dynamic_quant_asymmetric(layer_type, in_tensors, cur_param, expected_result):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}
    if layer_type == LayerNormType.LAYER_NORM_NROM.value:
        op.op_param['normParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
        op.op_param['postNormParam'] = cur_param

    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("layer_type, in_tensors, cur_param, expected_result", [
    (LayerNormType.LAYER_NORM_NROM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_TENSOR.value}, [torch.randn(2, 4)]),
    (LayerNormType.LAYER_NORM_POSTNORM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_TENSOR.value}, [torch.randn(2, 4)]),
])
def test_golden_calc_when_quant_type_per_tensor(layer_type, in_tensors, cur_param, expected_result):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}
    if layer_type == LayerNormType.LAYER_NORM_NROM.value:
        op.op_param['normParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
        op.op_param['postNormParam'] = cur_param

    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("layer_type, in_tensors, cur_param, expected_result", [
    (LayerNormType.LAYER_NORM_NROM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_CHANNEL.value}, [torch.randn(2, 4)]),
    (LayerNormType.LAYER_NORM_POSTNORM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_CHANNEL.value}, [torch.randn(2, 4)]),
])
def test_golden_calc_when_quant_type_per_channel(layer_type, in_tensors, cur_param, expected_result):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}
    if layer_type == LayerNormType.LAYER_NORM_NROM.value:
        op.op_param['normParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
        op.op_param['postNormParam'] = cur_param

    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("layer_type, in_tensors, cur_param, expected_result", [
    (LayerNormType.LAYER_NORM_NROM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_GROUP.value}, [torch.randn(2, 4)]),
    (LayerNormType.LAYER_NORM_POSTNORM.value, [torch.randn(2, 4), torch.randn(4), torch.randn(4), torch.randn(4)],
     {'epsilon': 1e-5, 'quantType': QuantType.QUANT_TYPE_PER_GROUP.value}, [torch.randn(2, 4)]),
])
def test_golden_calc_when_quant_type_per_group(layer_type, in_tensors, cur_param, expected_result):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}
    if layer_type == LayerNormType.LAYER_NORM_NROM.value:
        op.op_param['normParam'] = cur_param
    elif layer_type == LayerNormType.LAYER_NORM_POSTNORM.value:
        op.op_param['postNormParam'] = cur_param

    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)
