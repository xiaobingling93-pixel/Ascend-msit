from unittest.mock import patch
import pytest
import torch
import torch_npu
from msit_llm.opcheck.check_case.self_attention import OpcheckUnpadSelfAttentionOperation, CalcType, KvCacheCfg, \
    MaskType, KernelType, ClampType

# Mocking the OperationTest class to avoid errors
from mock_operation_test import MockOperationTest

# Use the new OperationTest class to replace the original OperationTest
OpcheckUnpadSelfAttentionOperation.__bases__ = (MockOperationTest,)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown


# Test cases for attention_mask_nz_2_nd
@pytest.mark.parametrize("attention_mask, seq_len, expected_shape", [
    (torch.randn(4, 8, 16, 32), [4, 8, 12, 16], (64, 8, 16)),
])
def test_attention_mask_nz_2_nd_when_valid_input_then_correct_shape(attention_mask, seq_len, expected_shape):
    result_mask = OpcheckUnpadSelfAttentionOperation.attention_mask_nz_2_nd(attention_mask, seq_len)
    assert result_mask.shape == expected_shape


# Test cases for group_matmul
@pytest.mark.parametrize("head_num, group_num, in_a, in_b, expected_shape", [
    (8, 4, torch.randn(8, 16, 32), torch.randn(4, 32, 16), (8, 16, 16)),
])
def test_group_matmul_when_valid_input_then_correct_shape(head_num, group_num, in_a, in_b, expected_shape):
    score = OpcheckUnpadSelfAttentionOperation.group_matmul(head_num, group_num, in_a, in_b)
    assert score.shape == expected_shape


@pytest.mark.parametrize("head_num, group_num, in_a, in_b", [
    (8, 0, torch.randn(8, 16, 32), torch.randn(4, 32, 16)),
])
def test_group_matmul_when_zero_division_then_raise_error(head_num, group_num, in_a, in_b):
    with pytest.raises(RuntimeError):
        OpcheckUnpadSelfAttentionOperation.group_matmul(head_num, group_num, in_a, in_b)


# Test cases for reshape_qkv
@pytest.mark.parametrize("qkv, is_pa, expected_shape", [
    (torch.randn(4, 8, 16, 32), True, (32, 512)),
    (torch.randn(4, 8, 16), True, (4, 128)),
])
def test_reshape_qkv_when_valid_input_then_correct_shape(qkv, is_pa, expected_shape):
    result_qkv = OpcheckUnpadSelfAttentionOperation.reshape_qkv(qkv, is_pa)
    assert result_qkv.shape == expected_shape


# Test cases for get_qkv
@pytest.mark.parametrize("in_tensors, expected_shapes", [
    ([torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32)],
     [(32, 512), (32, 512), (32, 512)]),
])
def test_get_qkv_when_valid_input_then_correct_shape(in_tensors, expected_shapes):
    q, k, v = OpcheckUnpadSelfAttentionOperation.get_qkv(in_tensors)
    assert q.shape == expected_shapes[0]
    assert k.shape == expected_shapes[1]
    assert v.shape == expected_shapes[2]


# Test cases for get_out_sub
@pytest.mark.parametrize("head_info, q_s, score, v_slice, expected_shape", [
    ([8, 16, 4], 4, torch.randn(8, 4, 16), torch.randn(4, 16, 16), (4, 8, 16)),
])
def test_get_out_sub_when_valid_input_then_correct_shape(head_info, q_s, score, v_slice, expected_shape):
    _p = None
    out_sub, _p = OpcheckUnpadSelfAttentionOperation.get_out_sub(head_info, q_s, score, v_slice, _p)
    assert out_sub.shape == expected_shape


# Test cases for get_mask
@patch.object(OpcheckUnpadSelfAttentionOperation, 'get_soc_version')
@pytest.mark.parametrize("soc_version, in_tensors, seq_len, expected_shape", [
    ("Ascend910B",
     [torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32)],
     [4, 8, 12, 16], (4, 8, 16, 32)),
])
def test_get_mask_when_soc_version_ascend910b_then_correct_shape(mock_get_soc_version, soc_version, in_tensors, seq_len,
                                                                 expected_shape):
    mock_get_soc_version.return_value = soc_version
    op = OpcheckUnpadSelfAttentionOperation()
    mask = op.get_mask(in_tensors, seq_len)
    assert mask.shape == expected_shape


# Test cases for get_batch_status
@pytest.mark.parametrize("in_tensors, seq_len, expected_batch_status", [
    ([torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32),
      [4, 8, 12, 16], range(4)], [4, 8, 12, 16], [0, 1, 2, 3]),
    ([torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32),
      [4, 8, 12, 16]], [4, 8, 12, 16], [0, 1, 2, 3]),
])
def test_get_batch_status_when_valid_input_then_correct_shape(in_tensors, seq_len, expected_batch_status):
    op = OpcheckUnpadSelfAttentionOperation()
    batch_status = op.get_batch_status(in_tensors, seq_len)
    assert list(batch_status) == expected_batch_status


# Test cases for get_post_mask_coff
@pytest.mark.parametrize("data_type, mask_type, expected_value", [
    (torch.float16, MaskType.MASK_TYPE_UNDEFINED.value, 1.0),
    (torch.bfloat16, MaskType.MASK_TYPE_ALIBI.value, 1.0),
    (torch.float32, MaskType.MASK_TYPE_ALIBI.value, 1.0),
    (torch.float32, MaskType.MASK_TYPE_UNDEFINED.value, -3e38),
])
def test_get_post_mask_coff_when_valid_input_then_correct_value(data_type, mask_type, expected_value):
    op = OpcheckUnpadSelfAttentionOperation()
    op.op_param['maskType'] = mask_type
    post_mask_coff = op.get_post_mask_coff(data_type)
    assert post_mask_coff == expected_value


# Test cases for get_attention_params
@pytest.mark.parametrize("q, q_scale, qk_scale, head_num, kv_head_num, expected_params", [
    (torch.randn(4, 8, 16), 2.0, 3.0, 4, 2, [2.0, 3.0, [4, 2, 2], torch.float32, 4]),
])
def test_get_attention_params_when_valid_input_then_correct_values(q, q_scale, qk_scale, head_num, kv_head_num,
                                                                   expected_params):
    op = OpcheckUnpadSelfAttentionOperation()
    op.op_param['qScale'] = q_scale
    op.op_param['qkScale'] = qk_scale
    op.op_param['headNum'] = head_num
    op.op_param['kvHeadNum'] = kv_head_num
    params = op.get_attention_params(q)
    assert params == expected_params


# Test cases for get_clamped_score
@pytest.mark.parametrize("score, clamp_type, clamp_min, clamp_max, expected_min, expected_max", [
    (torch.randn(4, 8, 16), ClampType.CLAMP_TYPE_MIN_MAX.value, -1.0, 1.0, -1.0, 1.0),
    (torch.randn(4, 8, 16), ClampType.CLAMP_TYPE_UNDEFINED.value, 0.0, 0.0, float('-inf'), float('inf')),
])
def test_get_clamped_score_when_valid_input_then_correct_values(score, clamp_type, clamp_min, clamp_max, expected_min,
                                                                expected_max):
    op = OpcheckUnpadSelfAttentionOperation()
    op.op_param['clampType'] = clamp_type
    op.op_param['clampMin'] = clamp_min
    op.op_param['clampMax'] = clamp_max
    clamped_score = op.get_clamped_score(score)
    assert torch.all(clamped_score >= expected_min)
    assert torch.all(clamped_score <= expected_max)


# Test cases for golden_calc
@patch.object(OpcheckUnpadSelfAttentionOperation, 'kv_bypass_golden_func')
@patch.object(OpcheckUnpadSelfAttentionOperation, 'pa_encoder_golden_func')
@patch.object(OpcheckUnpadSelfAttentionOperation, 'undefined_golden_func')
@pytest.mark.parametrize("kvcache_cfg, calc_type, in_tensors, mock_golden_func, expected_shape", [
    (KvCacheCfg.K_BYPASS_V_BYPASS.value, CalcType.UNDEFINED.value,
     [torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), [4, 8, 12, 16], [4, 8, 12, 16], [0, 1, 2, 3],
      [0]], 'kv_bypass_golden_func', (4, 8 * 2)),
    (KvCacheCfg.K_CACHE_V_CACHE.value, CalcType.PA_ENCODER.value,
     [torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), [4, 8, 12, 16]], 'pa_encoder_golden_func',
     (4, 4, 2)),
    (KvCacheCfg.K_CACHE_V_CACHE.value, CalcType.UNDEFINED.value,
     [torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16),
      [4, 8, 12, 16], [0, 1, 2, 3], [0]], 'undefined_golden_func', (4, 8 * 2)),
])
def test_golden_calc_when_valid_input_then_correct_shape(mock_undefined_golden_func, mock_pa_encoder_golden_func,
                                                         mock_kv_bypass_golden_func, kvcache_cfg, calc_type, in_tensors,
                                                         mock_golden_func, expected_shape):
    op = OpcheckUnpadSelfAttentionOperation()
    op.op_param['kvcacheCfg'] = kvcache_cfg
    op.op_param['calcType'] = calc_type
    mock_golden_func_map = {
        'kv_bypass_golden_func': mock_kv_bypass_golden_func,
        'pa_encoder_golden_func': mock_pa_encoder_golden_func,
        'undefined_golden_func': mock_undefined_golden_func,
    }
    mock_golden_func_map[mock_golden_func].return_value = torch.randn(*expected_shape)
    golden = op.golden_calc(in_tensors)
    assert golden[0].shape == expected_shape
