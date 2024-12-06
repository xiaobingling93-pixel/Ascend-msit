import pytest
import torch
from unittest.mock import patch, MagicMock
from msit_llm.opcheck.operation_test import OpcheckAsStridedOperation
from msit_llm.common.log import logger


# 辅助函数：创建 mock 的 case_info 字典
def create_mock_case_info(size=None, stride=None, offset=None):
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
    case_info = create_mock_case_info(size=size, stride=stride, offset=offset)
    with patch.object(OpcheckAsStridedOperation, '__init__', return_value=None):
        op = OpcheckAsStridedOperation()
        op.case_info = case_info
        op.op_param = case_info['op_param']
        op.tensor_path = case_info['tensor_path']
        op.pid = case_info['pid']
        op.in_tensors = []
        op.out_tensors = []
        op.bind_idx = []
        op.atb_rerun = case_info['atb_rerun']
        op.optimization_closed = case_info['optimization_closed']
        op.precision_standard = {}  # 设置必要的属性，避免 AttributeError

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
    case_info = create_mock_case_info(size=size, stride=stride, offset=offset)
    with patch.object(OpcheckAsStridedOperation, '__init__', return_value=None):
        op = OpcheckAsStridedOperation()
        op.case_info = case_info
        op.op_param = case_info['op_param']
        op.tensor_path = case_info['tensor_path']
        op.pid = case_info['pid']
        op.in_tensors = []
        op.out_tensors = []
        op.bind_idx = []
        op.atb_rerun = case_info['atb_rerun']
        op.optimization_closed = case_info['optimization_closed']
        op.precision_standard = {}  # 设置必要的属性，避免 AttributeError

        with patch.object(op, 'validate_param', return_value=expected_return) as mock_validate_param:
            with patch.object(op, 'execute') as mock_execute:
                with patch('msit_llm.common.log.logger') as mock_logger:
                    op.test()

        mock_validate_param.assert_called_once_with("size", "stride", "offset")
        if expected_return:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()