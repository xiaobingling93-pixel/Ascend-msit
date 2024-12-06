import pytest
import torch
from unittest.mock import patch, MagicMock
from msit_llm.opcheck.check_case import OpcheckAllReduceOperation
from msit_llm.common.log import logger


# 测试静态方法
@pytest.mark.parametrize("cal_func, in_tensors, expected_result", [
    (OpcheckAllReduceOperation.sum_cal, [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
     [torch.tensor([4.0, 6.0])]),
    (OpcheckAllReduceOperation.max_cal, [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
     [torch.tensor([3.0, 4.0])]),
    (OpcheckAllReduceOperation.min_cal, [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
     [torch.tensor([1.0, 2.0])]),
    (OpcheckAllReduceOperation.prod_cal, [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
     [torch.tensor([3.0, 8.0])])
])
def test_static_methods_given_valid_input_when_valid_then_pass(cal_func, in_tensors, expected_result):
    result = cal_func(in_tensors)
    assert torch.allclose(result[0], expected_result[0])


# 测试 golden_calc 方法
@pytest.mark.parametrize("all_reduce_type, in_tensors, expected_result", [
    ("sum", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([4.0, 6.0])]),
    ("max", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 4.0])]),
    ("min", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([1.0, 2.0])]),
    ("prod", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 8.0])])
])
def test_golden_calc_given_valid_input_when_valid_then_pass(all_reduce_type, in_tensors, expected_result):
    op = OpcheckAllReduceOperation()
    op.op_param = {"allReduceType": all_reduce_type}
    op.get_new_in_tensors = MagicMock(return_value=in_tensors)
    result = op.golden_calc(in_tensors)
    assert torch.allclose(result[0], expected_result[0])


# 测试 test_all_reduce 方法
@pytest.mark.parametrize("pid, all_reduce_type, rank, rank_root, rank_size, backend, expected_log_error, "
                         "expected_return", [
                             (None, "sum", 0, 0, 2, "hccl",
                              "Cannot get a valid pid, AllReduceOperation is not supported!", None),
                             (1234, "sum", 0, 0, 2, "hccl", None, True)
                         ])
def test_test_all_reduce_given_valid_input_when_valid_then_pass(pid, all_reduce_type, rank, rank_root, rank_size,
                                                                backend, expected_log_error, expected_return):
    op = OpcheckAllReduceOperation()
    op.pid = pid
    op.op_param = {"allReduceType": all_reduce_type, "rank": rank, "rankRoot": rank_root, "rankSize": rank_size,
                   "backend": backend}

    with patch.object(op, 'validate_param', return_value=expected_return) as mock_validate_param:
        with patch.object(op, 'execute') as mock_execute:
            with patch.object(logger, 'error') as mock_logger_error:
                with patch.object(logger, 'debug') as mock_logger_debug:
                    op.test_all_reduce()

    if expected_log_error:
        mock_logger_error.assert_called_once_with(expected_log_error)
    else:
        mock_validate_param.assert_called_once_with("allReduceType", "rank", "rankRoot", "rankSize")
        if expected_return:
            mock_execute.assert_called_once()
            mock_logger_debug.assert_any_call(f"backend: {backend}, allreduceType: {all_reduce_type}")
            mock_logger_debug.assert_any_call("env: {}".format(os.getenv("LCCL_DETERMINISTIC", "")))
            mock_logger_debug.assert_any_call("env: {}".format(os.getenv("HCCL_DETERMINISTIC", "")))
