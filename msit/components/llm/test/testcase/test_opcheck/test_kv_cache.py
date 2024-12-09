from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckKvCacheOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckKvCacheOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("in_tensors, expected_result", [
    ([torch.randn(2, 4, 3, 3)], [torch.zeros((2, 3, 3))]),
    ([torch.randn(2, 4, 2, 2)], [torch.zeros((1, 2, 2))]),
    ([torch.randn(2, 4, 3, 3)], [torch.zeros((3, 3, 3))]),
])
def test_golden_calc_when_valid_input(in_tensors, expected_result):
    # Arrange
    op = OpcheckKvCacheOperation()
    op.op_param = {}

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert torch.allclose(result[0], expected_result[0], atol=1e-4)


@pytest.mark.parametrize("in_tensors, expected_error", [
    ([torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), torch.tensor([0]),
      torch.zeros((1, 2, 2, 2), dtype=torch.float16), torch.tensor([2]), torch.tensor([2])], RuntimeError),
    ([torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.int8), torch.tensor([0]),
      torch.zeros((1, 2, 2, 2), dtype=torch.float16), torch.tensor([2]), torch.tensor([2])], RuntimeError),
])
def test_golden_calc_when_invalid_input(in_tensors, expected_error):
    # Arrange
    op = OpcheckKvCacheOperation()
    op.op_param = {}

    # Act & Assert
    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


def test_test_when_valid_input():
    # Arrange
    op = OpcheckKvCacheOperation()
    op.op_param = {}

    # Act
    with patch.object(op, 'execute_inplace') as mock_execute:
        op.test()

    # Assert
    mock_execute.assert_called_once()
