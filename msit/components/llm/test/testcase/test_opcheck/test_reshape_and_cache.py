from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case.reshape_and_cache import OpcheckReshapeAndCacheOperation, CompressType
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


# Define test parameters for golden_calc with valid inputs
@dataclass
class TestParams:
    compress_type: int
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    TestParams(CompressType.COMPRESS_TYPE_KVHEAD.value,
               [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3),
                torch.tensor([0, 1]), torch.tensor([1, 1]), torch.tensor([1, 1])],
               [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
    TestParams(CompressType.COMPRESS_TYPE_UNDEFINED.value,
               [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3),
                torch.tensor([0, 1])],
               [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
])
def test_golden_calc_when_valid_input(mock_logger, params):
    mock_info, mock_debug = mock_logger
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    result = op.golden_calc(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


# Define test parameters for golden_calc with invalid inputs
@dataclass
class InvalidTestParams:
    compress_type: int
    in_tensors: list
    expected_error: type


@pytest.mark.parametrize("params", [
    InvalidTestParams(CompressType.COMPRESS_TYPE_KVHEAD.value,
                      [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3)],
                      RuntimeError),
    InvalidTestParams(CompressType.COMPRESS_TYPE_UNDEFINED.value,
                      [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3)],
                      RuntimeError),
])
def test_golden_calc_when_invalid_input(mock_logger, params):
    mock_info, mock_debug = mock_logger
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    with pytest.raises(params.expected_error):
        op.golden_calc(params.in_tensors)


# Define test parameters for golden_func1 with valid inputs
@dataclass
class TestFunc1Params:
    compress_type: int
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    TestFunc1Params(CompressType.COMPRESS_TYPE_KVHEAD.value,
                    [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3),
                     torch.tensor([0, 1]), torch.tensor([1, 1]), torch.tensor([1, 1])],
                    [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
])
def test_golden_func1_when_valid_input(mock_logger, params):
    mock_info, mock_debug = mock_logger
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    result = op.golden_func1(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


# Define test parameters for golden_func2 with valid inputs
@dataclass
class TestFunc2Params:
    compress_type: int
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    TestFunc2Params(CompressType.COMPRESS_TYPE_UNDEFINED.value,
                    [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3),
                     torch.tensor([0, 1])],
                    [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
])
def test_golden_func2_when_valid_input(mock_logger, params):
    mock_info, mock_debug = mock_logger
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    result = op.golden_func2(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


# Define test parameters for golden_func3 with valid inputs
@dataclass
class TestFunc3Params:
    compress_type: int
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    TestFunc3Params(CompressType.COMPRESS_TYPE_UNDEFINED.value,
                    [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3),
                     torch.tensor([0, 1])],
                    [torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3)]),
])
def test_golden_func3_when_valid_input(mock_logger, params):
    mock_info, mock_debug = mock_logger
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    result = op.golden_func3(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


# Define test parameters for golden_calc with different compress types
@dataclass
class TestDifferentCompressTypesParams:
    compress_type: int
    in_tensors: list
    expected_result: list


@pytest.mark.parametrize("params", [
    TestDifferentCompressTypesParams(CompressType.COMPRESS_TYPE_KVHEAD.value,
                                     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3),
                                      torch.randn(2, 3, 3), torch.tensor([0, 1]), torch.tensor([1, 1]),
                                      torch.tensor([1, 1])],
                                     [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
    TestDifferentCompressTypesParams(CompressType.COMPRESS_TYPE_UNDEFINED.value,
                                     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3),
                                      torch.randn(2, 3, 3), torch.tensor([0, 1])],
                                     [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
    TestDifferentCompressTypesParams(CompressType.COMPRESS_TYPE_UNDEFINED.value,
                                     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3, 3),
                                      torch.randn(2, 3, 3, 3), torch.tensor([0, 1])],
                                     [torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3)]),
])
def test_golden_calc_when_valid_input_with_different_compress_types(mock_logger, params):
    mock_info, mock_debug = mock_logger
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": params.compress_type}
    result = op.golden_calc(params.in_tensors)
    for res, exp in zip(result, params.expected_result):
        assert torch.allclose(res, exp, atol=1e-4)
