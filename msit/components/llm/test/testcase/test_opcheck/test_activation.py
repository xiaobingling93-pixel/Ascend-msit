import pytest
import torch
import torch_npu
import torch.nn.functional as F
from msit_llm.opcheck.check_case.activation import ActivationType, GeLUMode, ActivationGolden
from msit_llm.common.log import logger


# 测试 ActivationType 枚举中的每个值
@pytest.mark.parametrize("activation_type, expected_value", [
    (ActivationType.ACTIVATION_UNDEFINED, 0),
    (ActivationType.ACTIVATION_RELU, 1),
    (ActivationType.ACTIVATION_GELU, 2),
    (ActivationType.ACTIVATION_FAST_GELU, 3),
    (ActivationType.ACTIVATION_SWISH, 4),
    (ActivationType.ACTIVATION_LOG, 5),
    (ActivationType.ACTIVATION_SWIGLU_FORWARD, 6),
    (ActivationType.ACTIVATION_SWIGLU_BACKWARD, 7),
    (ActivationType.ACTIVATION_MAX, 8)
])
def test_activation_type_given_valid_activation_type_when_valid_then_pass(activation_type, expected_value):
    assert activation_type.value == expected_value


# 测试 GeLUMode 枚举中的每个值
@pytest.mark.parametrize("gelu_mode, expected_value", [
    (GeLUMode.TANH_MODE, 0),
    (GeLUMode.NONE_MODE, 1)
])
def test_gelu_mode_given_valid_gelu_mode_when_valid_then_pass(gelu_mode, expected_value):
    assert gelu_mode.value == expected_value


# 测试 ActivationGolden 中的每个静态方法
import pytest
import torch
from msit_llm.opcheck.check_case import ActivationGolden, ActivationType, GeLUMode


# 测试 ActivationGolden 中的每个静态方法
@pytest.mark.parametrize("activation_func, in_tensors, gelu_mode, scale, dim, expected_result", [
    (ActivationGolden.relu_golden, torch.tensor([-1.0, 0.0, 1.0]), None, None, None, torch.tensor([0.0, 0.0, 1.0])),
    (ActivationGolden.gelu_golden, torch.tensor([-1.0, 0.0, 1.0]), GeLUMode.TANH_MODE.value, None, None,
     torch.tensor([-0.1588, 0.0, 0.8412])),
    (ActivationGolden.fast_gelu_golden, torch.tensor([-1.0, 0.0, 1.0]), None, None, None,
     torch.tensor([-0.1588, 0.0, 0.8412])),
    (ActivationGolden.swish_golden, torch.tensor([-1.0, 0.0, 1.0]), None, 1.0, None,
     torch.tensor([-0.2689, 0.0, 0.7311])),
    (ActivationGolden.log_golden, torch.tensor([0.1, 1.0, 10.0]), None, None, None,
     torch.tensor([-2.3026, 0.0, 2.3026])),
    (ActivationGolden.swigluforward_golden, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), None, None, 1,
     torch.tensor([[0.7311, 2.0], [3.0, 4.0]])),
    (ActivationGolden.swiglubackward_golden,
     [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.0, 2.0], [3.0, 4.0]])], None, None, 1,
     torch.tensor([[0.7311, 2.0], [3.0, 4.0]]))
])
def test_activation_golden_given_valid_input_when_valid_then_pass(activation_func, in_tensors, gelu_mode, scale, dim,
                                                                  expected_result):
    if gelu_mode is not None:
        result = activation_func(in_tensors, gelu_mode)
    elif scale is not None:
        result = activation_func(in_tensors, scale)
    elif dim is not None:
        result = activation_func(in_tensors, dim)
    else:
        result = activation_func(in_tensors, None)
    assert torch.allclose(result, expected_result, atol=1e-4)


# 辅助函数：初始化 case_info 字典
def create_case_info(activation_type, gelu_mode=None, scale=None, dim=None):
    case_info = {
        'op_id': '1',
        'op_name': 'ActivationOperation',
        'op_param': {'activationType': activation_type.value},
        'tensor_path': '',
        'pid': 1234,
        'atb_rerun': False,
        'optimization_closed': False,
        'res_detail': []
    }
    if gelu_mode is not None:
        case_info['op_param']['geluMode'] = gelu_mode
    if scale is not None:
        case_info['op_param']['scale'] = scale
    if dim is not None:
        case_info['op_param']['dim'] = dim
    return case_info

# 测试 golden_calc 方法
@pytest.mark.parametrize("activation_type, in_tensors, gelu_mode, scale, dim, expected_result", [
    (ActivationType.ACTIVATION_RELU, [torch.tensor([-1.0, 0.0, 1.0])], None, None, None, [torch.tensor([0.0, 0.0, 1.0])]),
    (ActivationType.ACTIVATION_GELU, [torch.tensor([-1.0, 0.0, 1.0])], GeLUMode.TANH_MODE.value, None, None, [torch.tensor([-0.1588, 0.0, 0.8412])]),
    (ActivationType.ACTIVATION_FAST_GELU, [torch.tensor([-1.0, 0.0, 1.0])], None, None, None, [torch.tensor([-0.1588, 0.0, 0.8412])]),
    (ActivationType.ACTIVATION_SWISH, [torch.tensor([-1.0, 0.0, 1.0])], None, 1.0, None, [torch.tensor([-0.2689, 0.0, 0.7311])]),
    (ActivationType.ACTIVATION_LOG, [torch.tensor([0.1, 1.0, 10.0])], None, None, None, [torch.tensor([-2.3026, 0.0, 2.3026])]),
    (ActivationType.ACTIVATION_SWIGLU_FORWARD, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], None, None, 1, [torch.tensor([[0.7311, 2.0], [3.0, 4.0]])]),
    (ActivationType.ACTIVATION_SWIGLU_BACKWARD, [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.0, 2.0], [3.0, 4.0]])], None, None, 1, [torch.tensor([[0.7311, 2.0], [3.0, 4.0]])])
])
def test_golden_calc_given_valid_input_when_valid_then_pass(activation_type, in_tensors, gelu_mode, scale, dim, expected_result):
    case_info = create_case_info(activation_type, gelu_mode, scale, dim)
    op = OpcheckActivationOperation(case_info=case_info)
    result = op.golden_calc(in_tensors)
    for r, e in zip(result, expected_result):
        assert torch.allclose(r, e, atol=1e-4)

# 测试 test 方法
def test_given_valid_input_when_valid_then_pass():
    case_info = create_case_info(ActivationType.ACTIVATION_RELU)
    op = OpcheckActivationOperation(case_info=case_info)
    op.test()
    assert op.op_param == case_info['op_param']