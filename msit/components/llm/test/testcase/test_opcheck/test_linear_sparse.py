from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckLinearSparseOperation
from msit_llm.common.log import logger
from mock_operation_test import MockOperationTest
from dataclasses import dataclass

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckLinearSparseOperation.__bases__ = (MockOperationTest,)


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger():
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug


@dataclass
class TestParams:
    transposeA: bool
    transposeB: bool
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    TestParams(False, True, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
    TestParams(True, False, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
    TestParams(False, False, [torch.randn(2, 3), torch.randn(3, 2), None, torch.tensor([2.0])], [torch.randn(2, 2)]),
    TestParams(False, True, [torch.randn(2, 3), torch.randn(1, 3, 2, 4), torch.randn(2)], [torch.randn(2, 2)]),
    TestParams(True, False, [torch.randn(2, 3, 4), torch.randn(4, 3), torch.randn(3)], [torch.randn(2, 3, 3)]),
    TestParams(False, False, [torch.randn(2, 3), torch.randn(1, 3, 2, 4)], [torch.randn(2, 2)]),
    TestParams(True, False, [torch.randn(2, 3, 4), torch.randn(4, 3)], [torch.randn(2, 3, 3)]),
])
def test_golden_calc_when_valid_input(mock_logger, params):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": params.transposeA, "transposeB": params.transposeB}
    result = op.golden_calc(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@dataclass
class InvalidTestParams:
    transposeA: bool
    transposeB: bool
    in_tensors: list
    expected_error: type


@pytest.mark.parametrize("params", [
    InvalidTestParams(False, True, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    InvalidTestParams(True, False, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    InvalidTestParams(False, False, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
])
def test_golden_calc_when_invalid_input(mock_logger, params):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": params.transposeA, "transposeB": params.transposeB}
    with pytest.raises(params.expected_error):
        op.golden_calc(params.in_tensors)


@dataclass
class TestExecuteParams:
    transposeA: bool
    transposeB: bool
    tilingK: int
    tilingN: int
    validate_param_return: bool
    expected_execute_call: bool


@pytest.mark.parametrize("params", [
    TestExecuteParams(True, False, 8, 8, True, True),
    TestExecuteParams(False, True, 4, 4, True, True),
    TestExecuteParams(True, True, 8, 8, True, True),
    TestExecuteParams(False, False, 4, 4, True, True),
    TestExecuteParams(True, False, 8, 8, False, False),
    TestExecuteParams(False, True, 4, 4, False, False),
])
def test_test_when_valid_input(mock_logger, params):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": params.transposeA, "transposeB": params.transposeB, "tilingK": params.tilingK,
                   "tilingN": params.tilingN}
    op.validate_param = mock_validate_param(params.validate_param_return)
    with patch.object(op, 'execute') as mock_execute:
        op.test()
    if params.expected_execute_call:
        mock_execute.assert_called_once()
        mock_logger[1].assert_called_once_with(
            f"tilingK: {params.tilingK}, tilingN: {params.tilingN} \nOnly 8 is supported!")
    else:
        mock_execute.assert_not_called()


@dataclass
class TestBiasDeqScaleParams:
    transposeA: bool
    transposeB: bool
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    TestBiasDeqScaleParams(False, True, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
    TestBiasDeqScaleParams(True, False, [torch.randn(2, 3), torch.randn(3, 2), torch.randn(2)], [torch.randn(2, 2)]),
    TestBiasDeqScaleParams(False, False, [torch.randn(2, 3), torch.randn(3, 2), None, torch.tensor([2.0])],
                           [torch.randn(2, 2)]),
    TestBiasDeqScaleParams(False, True, [torch.randn(2, 3), torch.randn(1, 3, 2, 4), torch.randn(2)],
                           [torch.randn(2, 2)]),
    TestBiasDeqScaleParams(True, False, [torch.randn(2, 3, 4), torch.randn(4, 3), torch.randn(3)],
                           [torch.randn(2, 3, 3)]),
    TestBiasDeqScaleParams(False, False, [torch.randn(2, 3), torch.randn(1, 3, 2, 4)], [torch.randn(2, 2)]),
    TestBiasDeqScaleParams(True, False, [torch.randn(2, 3, 4), torch.randn(4, 3)], [torch.randn(2, 3, 3)]),
])
def test_golden_calc_when_valid_input_with_bias_and_deq_scale(mock_logger, params):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": params.transposeA, "transposeB": params.transposeB}
    result = op.golden_calc(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)
