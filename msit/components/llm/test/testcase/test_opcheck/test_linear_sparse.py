import pytest
import torch
from unittest.mock import patch
from msit_llm.opcheck.check_case import OpcheckLinearSparseOperation
from msit_llm.common.log import logger
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckLinearSparseOperation.__bases__ = (MockOperationTest,)


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger():
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug


def mock_get_soc_version(soc_version):
    def _get_soc_version():
        return soc_version

    return _get_soc_version


def mock_convert_data_format(tensor):
    return tensor


@pytest.mark.parametrize("soc_version, transposeA, transposeB, in_tensors, expected_result", [
    ('Ascend310P', False, True, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
])
def test_golden_calc_when_valid_input(mock_logger, soc_version, transposeA, transposeB, in_tensors, expected_result):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": transposeA, "transposeB": transposeB}
    op.get_soc_version = mock_get_soc_version(soc_version)
    op.convert_data_format = mock_convert_data_format
    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)
    if soc_version == 'Ascend310P':
        mock_logger[0].assert_called_once()


@pytest.mark.parametrize("soc_version, transposeA, transposeB, in_tensors, expected_error", [
    ('Ascend310P', False, True, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
])
def test_golden_calc_when_invalid_input(soc_version, transposeA, transposeB, in_tensors, expected_error):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": transposeA, "transposeB": transposeB}
    op.get_soc_version = mock_get_soc_version(soc_version)
    op.convert_data_format = mock_convert_data_format
    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


@pytest.mark.parametrize("transposeA, transposeB, tilingK, tilingN, validate_param_return, expected_execute_call", [
    (True, False, 8, 8, True, True),
    (False, True, 4, 4, True, True),
    (True, True, 8, 8, True, True),
    (False, False, 4, 4, True, True),
    (True, False, 8, 8, False, False),
    (False, True, 4, 4, False, False),
])
def test_test_when_valid_input(mock_logger, transposeA, transposeB, tilingK, tilingN, validate_param_return,
                               expected_execute_call):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": transposeA, "transposeB": transposeB, "tilingK": tilingK, "tilingN": tilingN}
    op.validate_param = lambda *args: validate_param_return
    with patch.object(op, 'execute') as mock_execute:
        op.test()
    if expected_execute_call:
        mock_execute.assert_called_once()
        mock_logger[1].assert_called_once_with(f"tilingK: {tilingK}, tilingN: {tilingN} \nOnly 8 is supported!")
    else:
        mock_execute.assert_not_called()


@pytest.mark.parametrize("soc_version, transposeA, transposeB, in_tensors, expected_result", [
    ('Ascend310P', False, True, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
])
def test_golden_calc_when_valid_input_with_bias_and_deq_scale(mock_logger, soc_version, transposeA, transposeB,
                                                              in_tensors, expected_result):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": transposeA, "transposeB": transposeB}
    op.get_soc_version = mock_get_soc_version(soc_version)
    op.convert_data_format = mock_convert_data_format
    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)
    if soc_version == 'Ascend310P':
        mock_logger[0].assert_called_once()
