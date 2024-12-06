import pytest
import torch
from unittest.mock import patch, MagicMock
from msit_llm.opcheck.check_case import OpcheckAllGatherOperation
from msit_llm.common.log import logger


# 辅助函数：初始化 case_info 字典
def create_case_info(pid=None, rank=None, rank_root=None, rank_size=None):
    return {
        'op_id': '1',
        'op_name': 'AllGatherOperation',
        'op_param': {'rank': rank, 'rankRoot': rank_root, 'rankSize': rank_size},
        'tensor_path': '',
        'pid': pid,
        'atb_rerun': False,
        'optimization_closed': False,
        'res_detail': []
    }


# 测试 golden_calc 方法
@pytest.mark.parametrize("in_tensors, expected_result", [
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
    ([torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])],
     torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
])
def test_golden_calc_given_valid_input_when_valid_then_pass(in_tensors, expected_result):
    case_info = create_case_info(pid=1234)
    op = OpcheckAllGatherOperation(case_info=case_info)
    op.get_new_in_tensors = MagicMock(return_value=in_tensors)
    result = op.golden_calc(in_tensors)
    assert torch.allclose(result[0], expected_result)


# 测试 test_all_gather 方法
@pytest.mark.parametrize("pid, rank, rank_root, rank_size, expected_log_error, expected_return", [
    (None, None, None, None, "Cannot get a valid pid, AllGatherOperation is not supported!", None),
    (1234, 0, 0, 2, None, True)
])
def test_all_gather_given_valid_input_when_valid_then_pass(pid, rank, rank_root, rank_size, expected_log_error,
                                                           expected_return):
    case_info = create_case_info(pid=pid, rank=rank, rank_root=rank_root, rank_size=rank_size)
    op = OpcheckAllGatherOperation(case_info=case_info)

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
