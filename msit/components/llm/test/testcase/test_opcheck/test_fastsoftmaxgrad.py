from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckFastSoftMaxGradOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckFastSoftMaxGradOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_test_fastsoftmaxgrad_given_batch_size_head_num_seq_len_range_when_valid_input_then_execute_successfully(
        batch_size_imm, head_num_imm, seq_len_range):
    # Arrange
    op = OpcheckFastSoftMaxGradOperation()
    op.op_param = {}

    # Act
    with patch.object(op, 'execute') as mock_execute:
        op.test_fastsoftmaxgrad()

    # Assert
    mock_execute.assert_called_once()


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_golden_calc_given_batch_size_head_num_seq_len_range_when_valid_input_then_correct_result(batch_size_imm,
                                                                                                  head_num_imm,
                                                                                                  seq_len_range):
    # Arrange
    op = OpcheckFastSoftMaxGradOperation()
    op.op_param = {}

    # Act
    result = op.golden_calc(None)

    # Assert
    assert result[0].shape == (batch_size_imm * head_num_imm * seq_len_range[1], seq_len_range[1])


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_gen_softmax_grad_given_head_num_seq_len_when_valid_input_then_correct_result(batch_size_imm, head_num_imm,
                                                                                      seq_len_range):
    # Arrange
    head_num = head_num_imm
    seq_len = seq_len_range[1]

    # Act
    y, y_grad, x_grad = gen_softmax_grad(head_num, seq_len)

    # Assert
    assert y.shape == (head_num * seq_len, seq_len)
    assert y_grad.shape == (head_num * seq_len, seq_len)
    assert x_grad.shape == (head_num * seq_len, seq_len)


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_golden_calc_given_batch_size_head_num_seq_len_range_when_valid_input_then_correct_result(batch_size_imm,
                                                                                                  head_num_imm,
                                                                                                  seq_len_range):
    # Arrange
    op = OpcheckFastSoftMaxGradOperation()
    op.op_param = {}

    # Act
    result = op.golden_calc(None)

    # Assert
    assert result[0].shape == (batch_size_imm * head_num_imm * seq_len_range[1], seq_len_range[1])


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_test_fastsoftmaxgrad_given_batch_size_head_num_seq_len_range_when_valid_input_then_execute_successfully(
        batch_size_imm, head_num_imm, seq_len_range):
    # Arrange
    op = OpcheckFastSoftMaxGradOperation()
    op.op_param = {}

    # Act
    with patch.object(op, 'execute') as mock_execute:
        op.test_fastsoftmaxgrad()

    # Assert
    mock_execute.assert_called_once()


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_gen_softmax_grad_given_head_num_seq_len_when_valid_input_then_correct_result(batch_size_imm, head_num_imm,
                                                                                      seq_len_range):
    # Arrange
    head_num = head_num_imm
    seq_len = seq_len_range[1]

    # Act
    y, y_grad, x_grad = gen_softmax_grad(head_num, seq_len)

    # Assert
    assert y.shape == (head_num * seq_len, seq_len)
    assert y_grad.shape == (head_num * seq_len, seq_len)
    assert x_grad.shape == (head_num * seq_len, seq_len)
