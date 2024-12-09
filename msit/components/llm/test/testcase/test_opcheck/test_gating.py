from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckGatingOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckGatingOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_result", [
    ({'seqLen': [2, 3], 'headNum': 2}, [torch.randint(0, 10, (2, 4, 3, 3)).float()], [torch.randint(0, 10, (2, 3, 3)).float()]),
    ({'seqLen': [1, 2], 'headNum': 1}, [torch.randint(0, 10, (2, 4, 2, 2)).float()], [torch.randint(0, 10, (1, 2, 2)).float()]),
    ({'seqLen': [3, 3], 'headNum': 3}, [torch.randint(0, 10, (2, 4, 3, 3)).float()], [torch.randint(0, 10, (3, 3, 3)).float()]),
])
def test_golden_calc_when_valid_input(op_param, in_tensors, expected_result):
    # Arrange
    op = OpcheckGatingOperation()
    op.op_param = op_param

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert torch.allclose(result[0], expected_result[0], atol=1e-4)


@pytest.mark.parametrize("op_param, in_tensors, expected_error", [
    ({'topkExpertNum': 2, 'cumSumNum': 3}, [torch.tensor([0, 1, 2, 0, 1])], RuntimeError),
    ({'topkExpertNum': 1, 'cumSumNum': 2}, [torch.tensor([0, 1, 0])], RuntimeError),
    ({'topkExpertNum': 3, 'cumSumNum': 4}, [torch.tensor([0, 1, 2, 3, 0, 1, 2])], RuntimeError),
])
def test_golden_calc_when_invalid_input(op_param, in_tensors, expected_error):
    # Arrange
    op = OpcheckGatingOperation()
    op.op_param = op_param

    # Act & Assert
    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'topkExpertNum': 2, 'cumSumNum': 3}, True, True),
    ({'topkExpertNum': 1, 'cumSumNum': 2}, True, True),
    ({'topkExpertNum': 3, 'cumSumNum': 4}, True, True),
    ({'topkExpertNum': 2}, False, False),
    ({'cumSumNum': 3}, False, False),
    ({}, False, False),
])
def test_test_when_valid_input(op_param, validate_param_return, expected_execute_call):
    # Arrange
    op = OpcheckGatingOperation()
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
