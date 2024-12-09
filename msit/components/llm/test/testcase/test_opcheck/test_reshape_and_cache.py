from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckReshapeAndCacheOperation, CompressType
from msit_llm.common.log import logger
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckReshapeAndCacheOperation.__bases__ = (MockOperationTest,)


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger():
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug


@pytest.mark.parametrize("compress_type, in_tensors, expected_result", [
    (CompressType.COMPRESS_TYPE_KVHEAD.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3), torch.tensor([0, 1]),
      torch.tensor([1, 1]), torch.tensor([1, 1])], [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
    (CompressType.COMPRESS_TYPE_UNDEFINED.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3), torch.tensor([0, 1])],
     [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
])
def test_golden_calc_when_valid_input(mock_logger, compress_type, in_tensors, expected_result):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": compress_type}
    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("compress_type, in_tensors, expected_error", [
    (CompressType.COMPRESS_TYPE_KVHEAD.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3)], RuntimeError),
    (CompressType.COMPRESS_TYPE_UNDEFINED.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3)], RuntimeError),
])
def test_golden_calc_when_invalid_input(mock_logger, compress_type, in_tensors, expected_error):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": compress_type}
    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


@pytest.mark.parametrize("compress_type, in_tensors, expected_result", [
    (CompressType.COMPRESS_TYPE_KVHEAD.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3), torch.tensor([0, 1]),
      torch.tensor([1, 1]), torch.tensor([1, 1])], [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
])
def test_golden_func1_when_valid_input(mock_logger, compress_type, in_tensors, expected_result):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": compress_type}
    result = op.golden_func1(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("compress_type, in_tensors, expected_result", [
    (CompressType.COMPRESS_TYPE_UNDEFINED.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3), torch.tensor([0, 1])],
     [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
])
def test_golden_func2_when_valid_input(mock_logger, compress_type, in_tensors, expected_result):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": compress_type}
    result = op.golden_func2(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("compress_type, in_tensors, expected_result", [
    (CompressType.COMPRESS_TYPE_UNDEFINED.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3),
      torch.tensor([0, 1])], [torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3)]),
])
def test_golden_func3_when_valid_input(mock_logger, compress_type, in_tensors, expected_result):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": compress_type}
    result = op.golden_func3(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("compress_type, in_tensors, expected_result", [
    (CompressType.COMPRESS_TYPE_KVHEAD.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3), torch.tensor([0, 1]),
      torch.tensor([1, 1]), torch.tensor([1, 1])], [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
    (CompressType.COMPRESS_TYPE_UNDEFINED.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3), torch.randn(2, 3, 3), torch.tensor([0, 1])],
     [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]),
    (CompressType.COMPRESS_TYPE_UNDEFINED.value,
     [torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3),
      torch.tensor([0, 1])], [torch.randn(2, 3, 3, 3), torch.randn(2, 3, 3, 3)]),
])
def test_golden_calc_when_valid_input_with_different_compress_types(mock_logger, compress_type, in_tensors,
                                                                    expected_result):
    op = OpcheckReshapeAndCacheOperation()
    op.op_param = {"compressType": compress_type}
    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)
