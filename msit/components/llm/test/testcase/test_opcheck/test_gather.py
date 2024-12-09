from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckGatherOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckGatherOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_result", [
    ({'seqLen': [2, 3], 'headNum': 2}, [torch.randn(2, 4, 3, 3)], [torch.randn(2, 3, 3)]),
    ({'seqLen': [1, 2], 'headNum': 1}, [torch.randn(2, 4, 2, 2)], [torch.randn(1, 2, 2)]),
    ({'seqLen': [3, 3], 'headNum': 3}, [torch.randn(2, 4, 3, 3)], [torch.randn(3, 3, 3)]),
])
def test_golden_calc_when_valid_input(op_param, in_tensors, expected_result):
    # Arrange
    op = OpcheckGatherOperation()
    op.op_param = op_param

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert torch.allclose(result[0], expected_result[0], atol=1e-4)


@pytest.mark.parametrize("op_param, in_tensors, expected_error", [
    ({'axis': 0, 'batchDims': 0}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], RuntimeError),
    ({'axis': 1, 'batchDims': 0}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], RuntimeError),
    ({'axis': 0, 'batchDims': 1}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], RuntimeError),
    ({'axis': 1, 'batchDims': 1}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], RuntimeError),
    ({'axis': 0, 'batchDims': -1}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0, 1])], None),
])
def test_golden_calc_when_invalid_input(op_param, in_tensors, expected_error):
    # Arrange
    op = OpcheckGatherOperation()
    op.op_param = op_param

    # Act & Assert
    if expected_error:
        with pytest.raises(expected_error):
            op.golden_calc(in_tensors)
    else:
        result = op.golden_calc(in_tensors)
        assert torch.allclose(result[0], in_tensors[0], atol=1e-4)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'axis': 0, 'batchDims': 0}, True, True),
    ({'axis': 1, 'batchDims': 0}, True, True),
    ({'axis': 0, 'batchDims': 1}, True, True),
    ({'axis': 1, 'batchDims': 1}, True, True),
    ({'axis': 0}, False, False),
    ({'batchDims': 0}, False, False),
    ({}, False, False),
])
def test_test_when_valid_input(op_param, validate_param_return, expected_execute_call):
    # Arrange
    op = OpcheckGatherOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    # Assert
    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
