import pytest
import torch
import torch_npu
from msit_llm.opcheck.case_manager import OpcheckAsStridedOperation
from unittest.mock import patch

# Mocking the OperationTest class to avoid errors
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckAsStridedOperation.__bases__ = (MockOperationTest,)


def test_golden_calc_given_valid_params_when_valid_input_then_correct_shape():
    # Arrange
    op_param = {
        'size': [4, 8, 16],
        'stride': [128, 16, 1],
        'offset': [0]
    }
    in_tensors = [torch.randn(4, 8, 16)]
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (4, 8, 16)


def test_golden_calc_given_invalid_size_when_invalid_input_then_raise_error():
    # Arrange
    op_param = {
        'size': [4, 8, 16],
        'stride': [128, 16, 1],
        'offset': [0]
    }
    in_tensors = [torch.randn(4, 8, 15)]  # Incorrect shape
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act & Assert
    with pytest.raises(RuntimeError):
        op.golden_calc(in_tensors)


def test_golden_calc_given_missing_size_when_invalid_input_then_raise_error():
    # Arrange
    op_param = {
        'stride': [128, 16, 1],
        'offset': [0]
    }
    in_tensors = [torch.randn(4, 8, 16)]
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act & Assert
    with pytest.raises(KeyError):
        op.golden_calc(in_tensors)


def test_golden_calc_given_missing_stride_when_invalid_input_then_raise_error():
    # Arrange
    op_param = {
        'size': [4, 8, 16],
        'offset': [0]
    }
    in_tensors = [torch.randn(4, 8, 16)]
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act & Assert
    with pytest.raises(KeyError):
        op.golden_calc(in_tensors)


def test_golden_calc_given_missing_offset_when_invalid_input_then_raise_error():
    # Arrange
    op_param = {
        'size': [4, 8, 16],
        'stride': [128, 16, 1]
    }
    in_tensors = [torch.randn(4, 8, 16)]
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act & Assert
    with pytest.raises(KeyError):
        op.golden_calc(in_tensors)


def test_test_given_valid_params_when_valid_input_then_execute_successfully():
    # Arrange
    op_param = {
        'size': [4, 8, 16],
        'stride': [128, 16, 1],
        'offset': [0]
    }
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=True):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    # Assert
    mock_execute.assert_called_once()


def test_test_given_invalid_params_when_invalid_input_then_return_early():
    # Arrange
    op_param = {
        'size': [4, 8, 16],
        'stride': [128, 16, 1],
        'offset': [0]
    }
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=False):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    # Assert
    mock_execute.assert_not_called()


def test_test_given_missing_size_when_invalid_input_then_return_early():
    # Arrange
    op_param = {
        'stride': [128, 16, 1],
        'offset': [0]
    }
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=False):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    # Assert
    mock_execute.assert_not_called()


def test_test_given_missing_stride_when_invalid_input_then_return_early():
    # Arrange
    op_param = {
        'size': [4, 8, 16],
        'offset': [0]
    }
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=False):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    # Assert
    mock_execute.assert_not_called()


def test_test_given_missing_offset_when_invalid_input_then_return_early():
    # Arrange
    op_param = {
        'size': [4, 8, 16],
        'stride': [128, 16, 1]
    }
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=False):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    # Assert
    mock_execute.assert_not_called()
