from unittest.mock import patch, MagicMock

import pytest
import torch

from msit_llm.opcheck.check_case.activation import OpcheckReshapeAndCacheOperation, CompressType
from msit_llm.common.log import logger
from mock_operation_test import MockOperationTest
from dataclasses import dataclass

# Use the new OperationTest class to replace the original OperationTest
OpcheckReshapeAndCacheOperation.__bases__ = (MockOperationTest,)


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger():
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug


# Define test parameters for golden_func1 with different scenarios
@dataclass
class GoldenFunc1TestParams:
    compress_type: int
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    GoldenFunc1TestParams(
        compress_type=CompressType.COMPRESS_TYPE_KVHEAD.value,
        in_tensors=[
            torch.randn(2, 4, 3), torch.randn(2, 4, 3),
            torch.randn(2, 3, 3), torch.randn(2, 3, 3),
            torch.tensor([0, 1]), torch.tensor([1, 1]),
            torch.tensor([1, 1])
        ],
        expected_result=[torch.randn(2, 3, 3), torch.randn(2, 3, 3)]
    ),
    GoldenFunc1TestParams(
        compress_type=CompressType.COMPRESS_TYPE_KVHEAD.value,
        in_tensors=[
            torch.zeros(2, 4, 3), torch.zeros(2, 4, 3),
            torch.zeros(2, 3, 3), torch.zeros(2, 3, 3),
            torch.tensor([-1, 1]), torch.tensor([0, 0]),
            torch.tensor([0, 0])
        ],
        expected_result=[torch.zeros(2, 3, 3), torch.zeros(2, 3, 3)]
    ),
])
def test_golden_func1_when_valid_input(mock_logger, params):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    result = op.golden_func1(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


# Define test parameters for golden_func2 with different scenarios
@dataclass
class GoldenFunc2TestParams:
    compress_type: int
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    GoldenFunc2TestParams(
        compress_type=CompressType.COMPRESS_TYPE_UNDEFINED.value,
        in_tensors=[
            torch.randn(2, 4, 3), torch.randn(2, 4, 3),
            torch.randn(2, 3, 3), torch.randn(2, 3, 3),
            torch.tensor([0, 1])
        ],
        expected_result=[torch.randn(2, 3, 3), torch.randn(2, 3, 3)]
    ),
    GoldenFunc2TestParams(
        compress_type=CompressType.COMPRESS_TYPE_UNDEFINED.value,
        in_tensors=[
            torch.zeros(2, 4, 3), torch.zeros(2, 4, 3),
            torch.zeros(2, 3, 3), torch.zeros(2, 3, 3),
            torch.tensor([-1, -1])
        ],
        expected_result=[torch.zeros(2, 3, 3), torch.zeros(2, 3, 3)]
    ),
])
def test_golden_func2_when_valid_input(mock_logger, params):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    result = op.golden_func2(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


# Define test parameters for golden_func3 with different scenarios
@dataclass
class GoldenFunc3TestParams:
    compress_type: int
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    GoldenFunc3TestParams(
        compress_type=CompressType.COMPRESS_TYPE_UNDEFINED.value,
        in_tensors=[
            torch.randn(2, 4, 3), torch.randn(2, 4, 3),
            torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3),
            torch.tensor([0, 1])
        ],
        expected_result=[torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3)]
    ),
    GoldenFunc3TestParams(
        compress_type=CompressType.COMPRESS_TYPE_UNDEFINED.value,
        in_tensors=[
            torch.zeros(2, 4, 3), torch.zeros(2, 4, 3),
            torch.zeros(2, 3, 3, 3), torch.zeros(2, 3, 3, 3),
            torch.tensor([-1, -1])
        ],
        expected_result=[torch.zeros(2, 3, 3, 3), torch.zeros(2, 3, 3, 3)]
    ),
])
def test_golden_func3_when_valid_input(mock_logger, params):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    result = op.golden_func3(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)
