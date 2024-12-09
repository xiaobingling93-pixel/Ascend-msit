import pytest
import torch
from unittest.mock import patch

from msit_llm.opcheck.check_case.linear_parallel import OpcheckLinearParallelOperation, ParallelType, QuantType
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckLinearParallelOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'seqLen': [2, 3], 'headNum': 2}, True, True),
    ({'seqLen': [1, 2], 'headNum': 1}, True, True),
    ({'seqLen': [3, 3], 'headNum': 3}, True, True),
    ({'seqLen': [2, 3]}, False, False),
    ({'headNum': 2}, False, False),
    ({}, False, False),
])
def test_test_2d_half_when_valid_input(op_param, validate_param_return, expected_execute_call):
    # Arrange
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test_2d_half()

    # Assert
    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()


@pytest.mark.parametrize("in_tensors, op_param, expected_result", [
    ([torch.randn(2, 3), torch.randn(3, 2)], {'hasResidual': False}, torch.randn(2, 2)),
    ([torch.randn(2, 3), torch.randn(3, 2), torch.randn(2, 2)], {'hasResidual': True}, torch.randn(2, 2)),
])
def test_add_residual_when_valid_input(in_tensors, op_param, expected_result):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    result = op.add_residual(torch.randn(2, 2), in_tensors)
    assert torch.allclose(result, expected_result, atol=1e-4)


@pytest.mark.parametrize("in_tensors, op_param, expected_result", [
    ([torch.randn(2, 3), torch.randn(3, 2)], {'transWeight': True}, torch.randn(2, 2)),
    ([torch.randn(2, 3), torch.randn(2, 3)], {'transWeight': False}, torch.randn(2, 3)),
])
def test_get_matmul_result_when_valid_input(in_tensors, op_param, expected_result):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    result = op.get_matmul_result(in_tensors)
    assert torch.allclose(result, expected_result, atol=1e-4)


@pytest.mark.parametrize("in_tensors, op_param, quant_type, group_size, out_data_type, expected_result", [
    ([torch.randn(2, 3), torch.randn(3, 2)], {'transWeight': True}, QuantType.QUANT_TYPE_UNDEFINED.value, 0, 0, torch.randn(2, 2)),
    ([torch.randn(2, 3), torch.randn(3, 2), torch.randn(2, 2)], {'transWeight': True}, QuantType.QUANT_TYPE_PER_TENSOR.value, 0, 0, torch.randn(2, 2)),
    ([torch.randn(2, 3), torch.randn(3, 2), torch.randn(2, 2)], {'transWeight': True}, QuantType.QUANT_TYPE_PER_GROUP.value, 1, 0, torch.randn(2, 2)),
])
def test_get_quant_result_when_valid_input(in_tensors, op_param, quant_type, group_size, out_data_type, expected_result):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    result = op.get_quant_result(in_tensors, quant_type, group_size, out_data_type)
    assert torch.allclose(result, expected_result, atol=1e-4)


@pytest.mark.parametrize("in_tensors, op_param, quant_type, group_size, out_data_type, expected_result", [
    ([torch.randn(2, 3), torch.randn(3, 2)], {'transWeight': True}, QuantType.QUANT_TYPE_UNDEFINED.value, 0, 0, [torch.randn(2, 2)]),
    ([torch.randn(2, 3), torch.randn(3, 2)], {'transWeight': True}, QuantType.QUANT_TYPE_PER_TENSOR.value, 0, 0, [torch.randn(2, 2)]),
])
def test_pure_linear_when_valid_input(in_tensors, op_param, quant_type, group_size, out_data_type, expected_result):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    result = op.pure_linear(in_tensors, quant_type, group_size, out_data_type)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("in_tensors, op_param, rank_size, quant_type, group_size, out_data_type, expected_result", [
    ([torch.randn(2, 3), torch.randn(3, 2)], {'transWeight': True}, 2, QuantType.QUANT_TYPE_UNDEFINED.value, 0, 0, [torch.randn(2, 2)]),
    ([torch.randn(2, 3), torch.randn(3, 2)], {'transWeight': True}, 2, QuantType.QUANT_TYPE_PER_TENSOR.value, 0, 0, [torch.randn(2, 2)]),
])
def test_all_reduce_when_valid_input(in_tensors, op_param, rank_size, quant_type, group_size, out_data_type, expected_result):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    result = op.all_reduce(in_tensors, rank_size, quant_type, group_size, out_data_type)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("in_tensors, op_param, rank, rank_size, expected_result", [
    ([torch.randn(2, 3), torch.randn(3, 2)], {'transWeight': True}, 0, 2, [torch.randn(2, 2)]),
])
def test_reduce_scatter_when_valid_input(in_tensors, op_param, rank, rank_size, expected_result):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    result = op.reduce_scatter(in_tensors, rank, rank_size)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("in_tensors, op_param, rank_size, expected_result", [
    ([torch.randn(2, 3), torch.randn(3, 2)], {'transWeight': True}, 2, [torch.randn(2, 2)]),
])
def test_all_gather_linear_when_valid_input(in_tensors, op_param, rank_size, expected_result):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    result = op.all_gather_linear(in_tensors, rank_size)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("in_tensors, op_param, expected_result", [
    ([torch.randn(2, 3), torch.randn(3, 2)], {'backend': 'hccl', 'rank': 0, 'rankSize': 2}, [torch.randn(2, 2)]),
    ([torch.randn(2, 3), torch.randn(3, 2)], {'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.LINEAR_ALL_REDUCE.value}, [torch.randn(2, 2)]),
    ([torch.randn(2, 3), torch.randn(3, 2)], {'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.LINEAR_REDUCE_SCATTER.value}, [torch.randn(2, 2)]),
    ([torch.randn(2, 3), torch.randn(3, 2)], {'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.ALL_GATHER_LINEAR.value}, [torch.randn(2, 2)]),
    ([torch.randn(2, 3), torch.randn(3, 2)], {'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.PURE_LINEAR.value}, [torch.randn(2, 2)]),
])
def test_golden_calc_when_valid_input(in_tensors, op_param, expected_result):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    result = op.golden_calc(in_tensors)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("op_param, expected_execute_call", [
    ({'backend': 'hccl', 'rank': 0, 'rankSize': 2}, True),
    ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.LINEAR_ALL_REDUCE.value}, True),
    ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.LINEAR_REDUCE_SCATTER.value}, True),
    ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.ALL_GATHER_LINEAR.value}, True),
    ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.PURE_LINEAR.value}, True),
])
def test_test_when_valid_input(op_param, expected_execute_call):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    with patch.object(op, 'get_soc_version', return_value='Ascend910B'):
        with patch.object(op, 'validate_param', return_value=True):
            with patch.object(op, 'execute') as mock_execute:
                op.test()
                if expected_execute_call:
                    mock_execute.assert_called_once()
                else:
                    mock_execute.assert_not_called()