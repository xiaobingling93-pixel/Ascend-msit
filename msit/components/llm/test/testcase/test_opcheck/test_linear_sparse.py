from unittest.mock import patch

import pytest
import torch

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


def mock_validate_param(validate_param_return):
    def _validate_param(*args):
        return validate_param_return

    return _validate_param


@pytest.mark.parametrize("transposeA, transposeB, in_tensors, expected_result", [
    (False, True, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
    (True, False, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
    (False, False, [torch.randn(2, 3), torch.randn(3, 2), None, torch.tensor([2.0])], [torch.randn(2, 2)]),
    (False, True, [torch.randn(2, 3), torch.randn(1, 3, 2, 4), torch.randn(2)], [torch.randn(2, 2)]),
    (True, False, [torch.randn(2, 3, 4), torch.randn(4, 3), torch.randn(3)], [torch.randn(2, 3, 3)]),
    (False, False, [torch.randn(2, 3), torch.randn(1, 3, 2, 4)], [torch.randn(2, 2)]),
    (True, False, [torch.randn(2, 3, 4), torch.randn(4, 3)], [torch.randn(2, 3, 3)]),
])
def test_golden_calc_when_valid_input(mock_logger, transposeA, transposeB, in_tensors, expected_result):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": transposeA, "transposeB": transposeB}
    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("transposeA, transposeB, in_tensors, expected_error", [
    (False, True, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    (True, False, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    (False, False, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
])
def test_golden_calc_when_invalid_input(mock_logger, transposeA, transposeB, in_tensors, expected_error):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": transposeA, "transposeB": transposeB}
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
    op.validate_param = mock_validate_param(validate_param_return)
    with patch.object(op, 'execute') as mock_execute:
        op.test()
    if expected_execute_call:
        mock_execute.assert_called_once()
        mock_logger[1].assert_called_once_with(f"tilingK: {tilingK}, tilingN: {tilingN} \nOnly 8 is supported!")
    else:
        mock_execute.assert_not_called()


@pytest.mark.parametrize("transposeA, transposeB, in_tensors, expected_result", [
    (False, True, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
    (True, False, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
    (False, False, [torch.randn(2, 3), torch.randn(3, 2), None, torch.tensor([2.0])], [torch.randn(2, 2)]),
    (False, True, [torch.randn(2, 3), torch.randn(1, 3, 2, 4), torch.randn(2)], [torch.randn(2, 2)]),
    (True, False, [torch.randn(2, 3, 4), torch.randn(4, 3), torch.randn(3)], [torch.randn(2, 3, 3)]),
    (False, False, [torch.randn(2, 3), torch.randn(1, 3, 2, 4)], [torch.randn(2, 2)]),
    (True, False, [torch.randn(2, 3, 4), torch.randn(4, 3)], [torch.randn(2, 3, 3)]),
])
def test_golden_calc_when_valid_input_with_bias_and_deq_scale(mock_logger, transposeA, transposeB, in_tensors,
                                                              expected_result):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": transposeA, "transposeB": transposeB}
    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)
