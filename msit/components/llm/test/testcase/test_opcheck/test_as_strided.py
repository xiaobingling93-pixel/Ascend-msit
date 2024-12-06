import pytest
import torch
from unittest.mock import patch, MagicMock
from msit_llm.opcheck.check_case import OpcheckAsStridedOperation
from msit_llm.common.log import logger


# 辅助函数：初始化 case_info 字典
def create_case_info(size=None, stride=None, offset=None):
    case_info = {
        'op_id': '1',
        'op_name': 'AsStridedOperation',
        'op_param': {},
        'tensor_path': '',
        'pid': 1234,
        'atb_rerun': False,
        'optimization_closed': False,
        'res_detail': []
    }
    if size is not None:
        case_info['op_param']['size'] = size
    if stride is not None:
        case_info['op_param']['stride'] = stride
    if offset is not None:
        case_info['op_param']['offset'] = offset
    return case_info


# 测试 golden_calc 方法
@pytest.mark.parametrize("in_tensors, size, stride, offset, expected_result", [
    (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), [2, 2], [1, 2], [0], torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
    (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), [2, 1], [1, 2], [0], torch.tensor([[1.0], [3.0]])),
    (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), [1, 2], [2, 1], [0], torch.tensor([[1.0, 2.0]])),
    (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), [1, 1], [2, 2], [0], torch.tensor([[1.0]])),
])
def test_golden_calc_given_valid_input_when_valid_then_pass(in_tensors, size, stride, offset, expected_result):
    case_info = create_case_info(size=size, stride=stride, offset=offset)
    op = OpcheckAsStridedOperation(case_info=case_info)
    result = op.golden_calc([in_tensors])
    assert torch.allclose(result[0], expected_result)


# 测试 test 方法
@pytest.mark.parametrize("size, stride, offset, expected_return", [
    ([2, 2], [1, 2], [0], True),
    (None, [1, 2], [0], False),
    ([2, 2], None, [0], False),
    ([2, 2], [1, 2], None, False),
])
def test_test_given_valid_input_when_valid_then_pass(size, stride, offset, expected_return):
    case_info = create_case_info(size=size, stride=stride, offset=offset)
    op = OpcheckAsStridedOperation(case_info=case_info)

    with patch.object(op, 'validate_param', return_value=expected_return) as mock_validate_param:
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    mock_validate_param.assert_called_once_with("size", "stride", "offset")
    if expected_return:
        mock_execute.assert_called_once()
