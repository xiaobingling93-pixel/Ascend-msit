import pytest
import torch
from unittest.mock import patch, MagicMock
from msit_llm.opcheck.check_case import OpcheckAllGatherOperation
from msit_llm.common.log import logger


# 测试 OpcheckAllGatherOperation 的 golden_calc 方法
@pytest.mark.parametrize("in_tensors, expected_result", [
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
    ([torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])],
     torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
])
def test_golden_calc_given_valid_input_when_valid_then_pass(in_tensors, expected_result):
    op = OpcheckAllGatherOperation()
    op.get_new_in_tensors = MagicMock(return_value=in_tensors)
    result = op.golden_calc(in_tensors)
    assert torch.allclose(result[0], expected_result)


# 测试 OpcheckAllGatherOperation 的 test_all_gather 方法
@pytest.mark.parametrize("pid, rank, rank_root, rank_size, expected_log_error, expected_return", [
    (None, None, None, None, "Cannot get a valid pid, AllGatherOperation is not supported!", None),
    (1234, 0, 0, 2, None, True)
])
def test_all_gather_given_valid_input_when_valid_then_pass(pid, rank, rank_root,
                                                           rank_size,
                                                           expected_log_error,
                                                           expected_return):
    op = OpcheckAllGatherOperation()
    op.pid = pid
    op.op_param = {"rank": rank, "rankRoot": rank_root, "rankSize": rank_size}

    with patch.object(op, 'validate_param', return_value=expected_return) as mock_validate_param:
        with patch.object(op, 'execute') as mock_execute:
            with patch.object(logger, 'error') as mock_logger_error:
                op.test_all_gather()

    if expected_log_error:
        mock_logger_error.assert_called_once_with(expected_log_error)
    else:
        mock_validate_param.assert_called_once_with("rank", "rankRoot", "rankSize")
        if expected_return:
            mock_execute.assert_called_once()
