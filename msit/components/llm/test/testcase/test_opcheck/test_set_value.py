from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case.set_value import OpcheckSetValueOperation
# Mocking the OperationTest class to avoid errors
from mock_operation_test import MockOperationTest

# Using the new OperationTest class to replace the original OperationTest
OpcheckSetValueOperation.__bases__ = (MockOperationTest,)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown

def test_golden_calc_given_empty_starts_when_no_starts_then_return_original_tensors():
    # Arrange
    op = OpcheckSetValueOperation()
    op.op_param = {"starts": [], "ends": [], "strides": []}
    in_tensors = [torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1])]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert torch.equal(result[0], in_tensors[0])
    assert torch.equal(result[1], in_tensors[1])

def test_golden_calc_given_starts_ends_strides_when_ends_out_of_bounds_then_no_change():
    # Arrange
    op = OpcheckSetValueOperation()
    op.op_param = {"starts": [0], "ends": [10], "strides": [1]}  # End out of bounds
    in_tensors = [torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1])]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    expected_result = [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1, 1])]
    assert torch.equal(result[0], expected_result[0])
    assert torch.equal(result[1], expected_result[1])

class MockOperationTest:
    def __init__(self, *args, **kwargs):
        self.op_param = kwargs.get('op_param', {})
    
    @staticmethod
    def validate_param(self, *args):
        for param in args:
            if param not in self.op_param:
                return False
        return True
    
    @staticmethod
    def validate_int_range(self, value, valid_range, param_name):
        if value not in valid_range:
            raise ValueError(f"{param_name} must be in {valid_range}")
    
    def execute(self):
        # Mock execution; do nothing
        pass

# Replace the base class with the mock
OpcheckSetValueOperation.__bases__ = (MockOperationTest,)

def test_golden_calc_given_starts_ends_strides_when_valid_then_correct_output():
    # Arrange
    op = OpcheckSetValueOperation(op_param={'starts': [0], 'ends': [2], 'strides': [1]})
    in_tensor1 = torch.tensor([0, 0, 0])
    in_tensor2 = torch.tensor([1, 1])
    expected_output = in_tensor1.clone()
    expected_output[0:2:1] = in_tensor2
    # Act
    output = op.golden_calc([in_tensor1.clone(), in_tensor2.clone()])
    # Assert
    assert torch.equal(output[0], expected_output)
    assert torch.equal(output[1], in_tensor2)

def test_golden_calc_given_invalid_strides_when_invalid_then_raise_error():
    # Arrange
    op = OpcheckSetValueOperation(op_param={'starts': [0], 'ends': [2], 'strides': [2]})
    in_tensor1 = torch.tensor([0, 0, 0])
    in_tensor2 = torch.tensor([1, 1])
    in_tensors = [in_tensor1.clone(), in_tensor2.clone()]
    # Act & Assert
    with pytest.raises(ValueError, match="strides must be in"):
        op.golden_calc(in_tensors)

def test_golden_calc_given_starts_gt_ends_when_empty_slice_then_no_copy():
    # Arrange
    op = OpcheckSetValueOperation(op_param={'starts': [2], 'ends': [0], 'strides': [1]})
    in_tensor1 = torch.tensor([0, 0, 0])
    in_tensor2 = torch.tensor([1])
    expected_output = in_tensor1.clone()
    # Act
    output = op.golden_calc([in_tensor1.clone(), in_tensor2.clone()])
    # Assert
    assert torch.equal(output[0], expected_output)

def test_golden_calc_given_empty_lists_when_no_slicing_then_no_copy():
    # Arrange
    op = OpcheckSetValueOperation(op_param={'starts': [], 'ends': [], 'strides': []})
    in_tensor1 = torch.tensor([0, 0, 0])
    in_tensor2 = torch.tensor([1, 1])
    expected_output = in_tensor1.clone()
    # Act
    output = op.golden_calc([in_tensor1.clone(), in_tensor2.clone()])
    # Assert
    assert torch.equal(output[0], expected_output)

def test_golden_calc_given_tensor_when_valid_input_then_correct_copy_1d():
    op = OpcheckSetValueOperation()
    tensor_a = torch.tensor([0, 1, 2, 3, 4])
    tensor_b = torch.tensor([9, 8, 7])
    starts = [1]
    ends = [4]
    strides = [1]

    op.op_param = {
        "starts": starts,
        "ends": ends,
        "strides": strides
    }

    result = op.golden_calc([tensor_a, tensor_b])

    expected_result = torch.tensor([0, 9, 8, 7, 4])
    assert torch.equal(result[0], expected_result)

def test_golden_calc_given_tensor_when_strides_all_one_then_no_error():
    op = OpcheckSetValueOperation()
    tensor_a = torch.tensor([0, 1, 2, 3, 4])
    tensor_b = torch.tensor([9, 8, 7])
    starts = [1]
    ends = [4]
    strides = [1]

    op.op_param = {
        "starts": starts,
        "ends": ends,
        "strides": strides
    }

    try:
        result = op.golden_calc([tensor_a, tensor_b])
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError: {e}")

def test_golden_calc_given_tensor_when_invalid_stride_then_raise_error():
    op = OpcheckSetValueOperation()
    tensor_a = torch.tensor([0, 1, 2, 3, 4])
    tensor_b = torch.tensor([9, 8, 7])
    starts = [1]
    ends = [4]
    strides = [2]  # Invalid stride

    op.op_param = {
        "starts": starts,
        "ends": ends,
        "strides": strides
    }

    with pytest.raises(ValueError):
        op.golden_calc([tensor_a, tensor_b])

def test_golden_calc_given_tensor_when_single_element_tensor_then_correct_copy():
    op = OpcheckSetValueOperation()
    tensor_a = torch.tensor([0])
    tensor_b = torch.tensor([9])
    starts = [0]
    ends = [1]
    strides = [1]

    op.op_param = {
        "starts": starts,
        "ends": ends,
        "strides": strides
    }

    result = op.golden_calc([tensor_a, tensor_b])

    expected_result = torch.tensor([9])
    assert torch.equal(result[0], expected_result)

def test_golden_calc_given_tensor_when_empty_tensor_then_empty_result():
    op = OpcheckSetValueOperation()
    tensor_a = torch.empty((0,))
    tensor_b = torch.empty((0,))
    starts = [0]
    ends = [0]
    strides = [1]

    op.op_param = {
        "starts": starts,
        "ends": ends,
        "strides": strides
    }

    result = op.golden_calc([tensor_a, tensor_b])

    assert result[0].shape == torch.Size([0])

def test_golden_calc_given_tensor_when_mismatched_tensors_shape_then_raise_runtime_error():
    op = OpcheckSetValueOperation()
    tensor_a = torch.tensor([0, 1, 2, 3, 4])
    tensor_b = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1])
    starts = [1]
    ends = [4]
    strides = [1]

    op.op_param = {
        "starts": starts,
        "ends": ends,
        "strides": strides
    }

    with pytest.raises(RuntimeError):
        op.golden_calc([tensor_a, tensor_b])

def test_golden_calc_given_tensor_when_multiple_slices_then_correct_copy():
    op = OpcheckSetValueOperation()
    tensor_a = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tensor_b = torch.tensor([9, 8, 7])
    starts = [1, 5]
    ends = [4, 8]
    strides = [1, 1]

    op.op_param = {
        "starts": starts,
        "ends": ends,
        "strides": strides
    }

    result = op.golden_calc([tensor_a, tensor_b])

    expected_result = torch.tensor([0, 9, 8, 7, 4, 9, 8, 7, 8, 9])
    assert torch.equal(result[0], expected_result)

def test_golden_calc_given_tensor_when_tensor_b_larger_than_slice_then_raise_runtime_error():
    op = OpcheckSetValueOperation()
    tensor_a = torch.tensor([0, 1, 2, 3, 4])
    tensor_b = torch.tensor([9, 8, 7, 6])
    starts = [1]
    ends = [4]
    strides = [1]

    op.op_param = {
        "starts": starts,
        "ends": ends,
        "strides": strides
    }

    with pytest.raises(RuntimeError):
        op.golden_calc([tensor_a, tensor_b])