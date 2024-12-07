import pytest
import torch
import torch_npu
from msit_llm.opcheck.check_case.self_attention import OpcheckUnpadSelfAttentionOperation, CalcType, KvCacheCfg, MaskType, KernelType, ClampType
from unittest.mock import patch

# Mocking the OperationTest class to avoid errors
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckUnpadSelfAttentionOperation.__bases__ = (MockOperationTest,)

def test_attention_mask_nz_2_nd_given_attention_mask_seq_len_when_valid_input_then_correct_shape():
    # Arrange
    attention_mask = torch.randn(4, 8, 16, 32)
    seq_len = [4, 8, 12, 16]

    # Act
    result_mask = OpcheckUnpadSelfAttentionOperation.attention_mask_nz_2_nd(attention_mask, seq_len)

    # Assert
    assert result_mask.shape == (64, 8, 16)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown

# Test cases for group_matmul
def test_group_matmul_given_head_num_group_num_in_a_in_b_when_valid_input_then_correct_shape():
    head_num = 8
    group_num = 4
    in_a = torch.randn(8, 16, 32)
    in_b = torch.randn(4, 32, 16)
    score = OpcheckUnpadSelfAttentionOperation.group_matmul(head_num, group_num, in_a, in_b)
    assert score.shape == (8, 16, 16)

def test_group_matmul_given_head_num_group_num_in_a_in_b_when_zero_division_then_raise_error():
    head_num = 8
    group_num = 0
    in_a = torch.randn(8, 16, 32)
    in_b = torch.randn(4, 32, 16)
    with pytest.raises(RuntimeError):
        OpcheckUnpadSelfAttentionOperation.group_matmul(head_num, group_num, in_a, in_b)

# Test cases for attention_mask_nz_2_nd
def test_attention_mask_nz_2_nd_given_attention_mask_seq_len_when_valid_input_then_correct_shape():
    attention_mask = torch.randn(4, 8, 16, 32)
    seq_len = [4, 8, 12, 16]
    result_mask = OpcheckUnpadSelfAttentionOperation.attention_mask_nz_2_nd(attention_mask, seq_len)
    assert result_mask.shape == (64, 8, 16)


# Test cases for reshape_qkv
def test_reshape_qkv_given_qkv_is_pa_when_valid_input_then_correct_shape():
    qkv = torch.randn(4, 8, 16, 32)
    result_qkv = OpcheckUnpadSelfAttentionOperation.reshape_qkv(qkv, True)
    assert result_qkv.shape == (32, 512)

def test_reshape_qkv_given_qkv_is_pa_when_3d_input_then_correct_shape():
    qkv = torch.randn(4, 8, 16)
    result_qkv = OpcheckUnpadSelfAttentionOperation.reshape_qkv(qkv, True)
    assert result_qkv.shape == (4, 128)

# Test cases for get_qkv
def test_get_qkv_given_in_tensors_when_valid_input_then_correct_shape():
    in_tensors = [torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32)]
    q, k, v = OpcheckUnpadSelfAttentionOperation.get_qkv(in_tensors)
    assert q.shape == (32, 512)
    assert k.shape == (32, 512)
    assert v.shape == (32, 512)

# Test cases for get_out_sub
def test_get_out_sub_given_head_info_q_s_score_v_slice_p_when_valid_input_then_correct_shape():
    head_info = [8, 16, 4]
    q_s = 4
    score = torch.randn(8, 4, 16)
    v_slice = torch.randn(4, 16, 16)
    _p = None
    out_sub, _p = OpcheckUnpadSelfAttentionOperation.get_out_sub(head_info, q_s, score, v_slice, _p)
    assert out_sub.shape == (4, 8, 16)


@patch.object(OpcheckUnpadSelfAttentionOperation, 'get_soc_version')
def test_get_mask_given_in_tensors_seq_len_when_soc_version_ascend910b_then_correct_shape(mock_get_soc_version):
    mock_get_soc_version.return_value = "Ascend910B"
    op = OpcheckUnpadSelfAttentionOperation()
    in_tensors = [torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32)]
    seq_len = [4, 8, 12, 16]
    mask = op.get_mask(in_tensors, seq_len)
    assert mask.shape == (4, 8, 16, 32)

# Test cases for get_batch_status
def test_get_batch_status_given_in_tensors_seq_len_when_batch_run_status_enable_then_correct_shape():
    op = OpcheckUnpadSelfAttentionOperation()
    in_tensors = [torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), [4, 8, 12, 16], range(4)]
    seq_len = [4, 8, 12, 16]
    batch_status = op.get_batch_status(in_tensors, seq_len)
    assert list(batch_status) == [0, 1, 2, 3]

def test_get_batch_status_given_in_tensors_seq_len_when_batch_run_status_disable_then_correct_shape():
    op = OpcheckUnpadSelfAttentionOperation()
    in_tensors = [torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), torch.randn(4, 8, 16, 32), [4, 8, 12, 16]]
    seq_len = [4, 8, 12, 16]
    batch_status = op.get_batch_status(in_tensors, seq_len)
    assert list(batch_status) == [0, 1, 2, 3]

# Test cases for get_post_mask_coff
def test_get_post_mask_coff_given_data_type_when_kernel_type_high_precision_then_correct_value():
    op = OpcheckUnpadSelfAttentionOperation()
    data_type = torch.float16
    post_mask_coff = op.get_post_mask_coff(data_type)
    assert post_mask_coff == 1.0

def test_get_post_mask_coff_given_data_type_when_data_type_float16_then_correct_value():
    op = OpcheckUnpadSelfAttentionOperation()
    data_type = torch.float16
    post_mask_coff = op.get_post_mask_coff(data_type)
    assert post_mask_coff == 1.0

def test_get_post_mask_coff_given_data_type_when_data_type_bfloat16_and_is_alibi_then_correct_value():
    op = OpcheckUnpadSelfAttentionOperation()
    data_type = torch.bfloat16
    op.op_param['maskType'] = MaskType.MASK_TYPE_ALIBI.value
    post_mask_coff = op.get_post_mask_coff(data_type)
    assert post_mask_coff == 1.0

def test_get_post_mask_coff_given_data_type_when_data_type_float32_and_is_alibi_then_correct_value():
    op = OpcheckUnpadSelfAttentionOperation()
    data_type = torch.float32
    op.op_param['maskType'] = MaskType.MASK_TYPE_ALIBI.value
    post_mask_coff = op.get_post_mask_coff(data_type)
    assert post_mask_coff == 1.0

def test_get_post_mask_coff_given_data_type_when_data_type_float32_and_not_is_alibi_then_correct_value():
    op = OpcheckUnpadSelfAttentionOperation()
    data_type = torch.float32
    op.op_param['maskType'] = MaskType.MASK_TYPE_UNDEFINED.value
    post_mask_coff = op.get_post_mask_coff(data_type)
    assert post_mask_coff == -3e38

# Test cases for get_attention_params
def test_get_attention_params_given_q_when_valid_input_then_correct_values():
    op = OpcheckUnpadSelfAttentionOperation()
    q = torch.randn(4, 8, 16)
    op.op_param['qScale'] = 2.0
    op.op_param['qkScale'] = 3.0
    op.op_param['headNum'] = 4
    op.op_param['kvHeadNum'] = 2
    params = op.get_attention_params(q)
    assert params == [2.0, 3.0, [4, 2, 2], torch.float32, 4]

# Test cases for get_clamped_score
def test_get_clamped_score_given_score_when_clamp_type_min_max_then_correct_values():
    op = OpcheckUnpadSelfAttentionOperation()
    score = torch.randn(4, 8, 16)
    op.op_param['clampType'] = ClampType.CLAMP_TYPE_MIN_MAX.value
    op.op_param['clampMin'] = -1.0
    op.op_param['clampMax'] = 1.0
    clamped_score = op.get_clamped_score(score)
    assert torch.all(clamped_score >= -1.0)
    assert torch.all(clamped_score <= 1.0)

def test_get_clamped_score_given_score_when_clamp_type_undefined_then_correct_values():
    op = OpcheckUnpadSelfAttentionOperation()
    score = torch.randn(4, 8, 16)
    op.op_param['clampType'] = ClampType.CLAMP_TYPE_UNDEFINED.value
    clamped_score = op.get_clamped_score(score)
    assert torch.all(clamped_score == score)


# Test cases for golden_calc
@patch.object(OpcheckUnpadSelfAttentionOperation, 'kv_bypass_golden_func')
@patch.object(OpcheckUnpadSelfAttentionOperation, 'pa_encoder_golden_func')
@patch.object(OpcheckUnpadSelfAttentionOperation, 'undefined_golden_func')
def test_golden_calc_given_in_tensors_when_kvcache_cfg_k_bypass_v_bypass_then_correct_shape(mock_undefined_golden_func, mock_pa_encoder_golden_func, mock_kv_bypass_golden_func):
    op = OpcheckUnpadSelfAttentionOperation()
    in_tensors = [torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), [4, 8, 12, 16], [4, 8, 12, 16], [0, 1, 2, 3], [0]]
    op.op_param['kvcacheCfg'] = KvCacheCfg.K_BYPASS_V_BYPASS.value
    mock_kv_bypass_golden_func.return_value = torch.randn(4, 8 * 2)
    golden = op.golden_calc(in_tensors)
    assert golden[0].shape == (4, 8 * 2)

@patch.object(OpcheckUnpadSelfAttentionOperation, 'kv_bypass_golden_func')
@patch.object(OpcheckUnpadSelfAttentionOperation, 'pa_encoder_golden_func')
@patch.object(OpcheckUnpadSelfAttentionOperation, 'undefined_golden_func')
def test_golden_calc_given_in_tensors_when_calc_type_pa_encoder_then_correct_shape(mock_undefined_golden_func, mock_pa_encoder_golden_func, mock_kv_bypass_golden_func):
    op = OpcheckUnpadSelfAttentionOperation()
    in_tensors = [torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), [4, 8, 12, 16]]
    op.op_param['calcType'] = CalcType.PA_ENCODER.value
    mock_pa_encoder_golden_func.return_value = torch.randn(4, 4, 2)
    golden = op.golden_calc(in_tensors)
    assert golden[0].shape == (4, 4, 2)

@patch.object(OpcheckUnpadSelfAttentionOperation, 'kv_bypass_golden_func')
@patch.object(OpcheckUnpadSelfAttentionOperation, 'pa_encoder_golden_func')
@patch.object(OpcheckUnpadSelfAttentionOperation, 'undefined_golden_func')
def test_golden_calc_given_in_tensors_when_calc_type_undefined_then_correct_shape(mock_undefined_golden_func, mock_pa_encoder_golden_func, mock_kv_bypass_golden_func):
    op = OpcheckUnpadSelfAttentionOperation()
    in_tensors = [torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16), [4, 8, 12, 16], [0, 1, 2, 3], [0]]
    op.op_param['calcType'] = CalcType.UNDEFINED.value
    mock_undefined_golden_func.return_value = torch.randn(4, 8 * 2)
    golden = op.golden_calc(in_tensors)
    assert golden[0].shape == (4, 8 * 2)