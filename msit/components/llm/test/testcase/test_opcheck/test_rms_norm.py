import pytest
import torch
from msit_llm.opcheck.check_case.rms_norm import OpcheckRmsNormOperation, RmsNormType, QuantType, DynamicQuantType

from mock_operation_test import MockOperationTest

OpcheckRmsNormOperation.__bases__ = (MockOperationTest,)


def test_validate_rmsnorm_param_given_valid_layer_type_when_validate_then_correct_result():
    # Arrange
    op = OpcheckRmsNormOperation()
    op.op_param = {'layerType': RmsNormType.RMS_NORM_NORM.value, 'normParam': {'quantType': QuantType.QUANT_INT8.value}}

    # Act
    result = op.validate_rmsnorm_param(RmsNormType.RMS_NORM_NORM.value)

    # Assert
    assert result['quantType'] == QuantType.QUANT_INT8.value


@pytest.mark.parametrize("golden_output, beta, scale, offset, expected_result", [
    (torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)),
])
def test_rms_norm_quant_with_tensor_when_valid_input(golden_output, beta, scale, offset, expected_result):
    result = OpcheckRmsNormOperation.rms_norm_quant_with_tensor(golden_output, beta, scale, offset)
    assert torch.allclose(result, expected_result, atol=1e-4)


@pytest.mark.parametrize("golden_output, in_tensors, cur_param, expected_result", [
    (torch.randn(2, 3), [torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'dynamicQuantType': DynamicQuantType.DYNAMIC_QUANT_UNDEFINED.value}, [torch.randn(2, 3)]),
    (torch.randn(2, 3), [torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'dynamicQuantType': DynamicQuantType.DYNAMIC_QUANT_SYMMETRIC.value}, [torch.randn(2, 3), torch.randn(2, 3)]),
    (torch.randn(2, 3), [torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'dynamicQuantType': DynamicQuantType.DYNAMIC_QUANT_ASYMMETRIC.value},
     [torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)]),
])
def test_rms_norm_quant_when_valid_input(golden_output, in_tensors, cur_param, expected_result):
    op = OpcheckRmsNormOperation()
    result = op.rms_norm_quant(golden_output, in_tensors, cur_param)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("in_tensors, cur_param, layer_type, golden_output, x, expected_result", [
    ([torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'quantType': QuantType.QUANT_INT8.value}, RmsNormType.RMS_NORM_PRE_NORM.value, torch.randn(2, 3),
     torch.randn(2, 3), [torch.randn(2, 3), torch.randn(2, 3)]),
    ([torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'quantType': QuantType.QUANT_INT8.value}, RmsNormType.RMS_NORM_NORM.value, torch.randn(2, 3), torch.randn(2, 3),
     [torch.randn(2, 3)]),
    ([torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'quantType': QuantType.QUANT_UNDEFINED.value}, RmsNormType.RMS_NORM_PRE_NORM.value, torch.randn(2, 3),
     torch.randn(2, 3), [torch.randn(2, 3), torch.randn(2, 3)]),
    ([torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'quantType': QuantType.QUANT_UNDEFINED.value}, RmsNormType.RMS_NORM_NORM.value, torch.randn(2, 3),
     torch.randn(2, 3), [torch.randn(2, 3)]),
])
def test_get_golden_result_when_valid_input(in_tensors, cur_param, layer_type, golden_output, x, expected_result):
    op = OpcheckRmsNormOperation()
    result = op.get_golden_result(in_tensors, cur_param, layer_type, golden_output, x)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("in_tensors, op_param, expected_result", [
    ([torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'layerType': RmsNormType.RMS_NORM_NORM.value,
      'normParam': {'quantType': QuantType.QUANT_INT8.value, 'epsilon': 1e-5}}, [torch.randn(2, 3)]),
    ([torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'layerType': RmsNormType.RMS_NORM_PRE_NORM.value,
      'preNormParam': {'quantType': QuantType.QUANT_INT8.value, 'epsilon': 1e-5}},
     [torch.randn(2, 3), torch.randn(2, 3)]),
    ([torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)],
     {'layerType': RmsNormType.RMS_NORM_POST_NORM.value,
      'postNormParam': {'quantType': QuantType.QUANT_UNDEFINED.value, 'epsilon': 1e-5}}, [torch.randn(2, 3)]),
])
def test_golden_calc_when_valid_input(in_tensors, op_param, expected_result):
    op = OpcheckRmsNormOperation()
    op.op_param = op_param
    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("op_param, expected_execute_call", [
    ({'layerType': RmsNormType.RMS_NORM_NORM.value,
      'normParam': {'quantType': QuantType.QUANT_INT8.value, 'epsilon': 1e-5}}, True),
    ({'layerType': RmsNormType.RMS_NORM_PRE_NORM.value,
      'preNormParam': {'quantType': QuantType.QUANT_INT8.value, 'epsilon': 1e-5}}, True),
    ({'layerType': RmsNormType.RMS_NORM_POST_NORM.value,
      'postNormParam': {'quantType': QuantType.QUANT_UNDEFINED.value, 'epsilon': 1e-5}}, True),
])
def test_test_when_valid_input(op_param, expected_execute_call):
    op = OpcheckRmsNormOperation()
    op.op_param = op_param
    with patch.object(op, 'validate_param', return_value=True):
        with patch.object(op, 'execute') as mock_execute:
            op.test()
            if expected_execute_call:
                mock_execute.assert_called_once()
            else:
                mock_execute.assert_not_called()
