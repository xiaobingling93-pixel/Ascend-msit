import pytest
import torch
from msit_llm.opcheck.check_case.linear_sparse import OpcheckLinearSparseOperation

from mock_operation_test import MockOperationTest


OpcheckLinearSparseOperation.__bases__ = (MockOperationTest,)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown


def test_golden_calc_given_in_tensors_with_transposeB_true_when_2d_weight_then_transposed():
    # Arrange
    op = OpcheckLinearSparseOperation()
    op.op_param = {}