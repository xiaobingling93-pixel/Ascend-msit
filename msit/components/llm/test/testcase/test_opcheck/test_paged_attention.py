import pytest
import torch
import torch_npu
from msit_llm.opcheck.check_case.paged_attention import OpcheckPagedAttentionAttentionOperation, MaskType, QuantType, CompressType, CalcType
import pytest
import torch
import torch_npu
from msit_llm.opcheck.check_case.paged_attention import OpcheckPagedAttentionAttentionOperation, MaskType
from msit_llm.opcheck import operation_test

# Mocking the OperationTest class to avoid errors
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckPagedAttentionAttentionOperation.__bases__ = (MockOperationTest,)

from unittest.mock import patch

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown

# Test cases for mask_nz_2_nd
def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_alibi_then_correct_shape():
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.MASK_TYPE_ALIBI.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (1, head_num, 16, 16 * 16)

def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_norm_then_correct_shape():
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.MASK_TYPE_NORM.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (4, 16, 16 * 16)

def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_undefined_then_correct_shape():
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.UNDEFINED.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (4, 16, 16 * 16)

# Test cases for get_fp32_b
# def test_get_fp32_b_given_i_b_fp32_input_is_k_has_bias_when_is_k_true_then_correct_shape():
#     i = 0
#     b = torch.randn(4, 8, 16, 32)
#     fp32_input = [torch.randn(16), torch.randn(16), torch.randn(16), torch.randn(16)]
#     is_k = True
#     has_bias = True
#     fp32_b = OpcheckPagedAttentionAttentionOperation.get_fp32_b(i, b, fp32_input, is_k, has_bias)
#     assert fp32_b.shape == (8, 32, 16)

# def test_get_fp32_b_given_i_b_fp32_input_is_k_has_bias_when_is_k_false_then_correct_shape():
#     i = 0
#     b = torch.randn(4, 8, 16, 32)
#     fp32_input = [torch.randn(16), torch.randn(16), torch.randn(16), torch.randn(16)]
#     is_k = False
#     has_bias = True
#     fp32_b = OpcheckPagedAttentionAttentionOperation.get_fp32_b(i, b, fp32_input, is_k, has_bias)
#     assert fp32_b.shape == (8, 32, 16)

# # Test cases for get_quant_param
def test_get_quant_param_given_quant_type_has_quant_offset_when_quant_type_undefined_then_correct_values():
    op = OpcheckPagedAttentionAttentionOperation()
    quant_type = QuantType.TYPE_QUANT_UNDEFINED.value
    has_quant_offset = False
    is_int8_flag, has_bias, fp32_input = op.get_quant_param(quant_type, has_quant_offset)
    assert is_int8_flag == False
    assert has_bias == False
    assert fp32_input == [None, None, None, None]

# def test_get_quant_param_given_quant_type_has_quant_offset_when_quant_type_dequant_fusion_then_correct_values():
#     op = OpcheckPagedAttentionAttentionOperation()
#     quant_type = QuantType.TYPE_DEQUANT_FUSION.value
#     has_quant_offset = True
#     is_int8_flag, has_bias, fp32_input = op.get_quant_param(quant_type, has_quant_offset)
#     assert is_int8_flag == True
#     assert has_bias == True
#     assert len(fp32_input) == 4

# # Test cases for group_matmul
# @patch.object(OpcheckPagedAttentionAttentionOperation, 'get_quant_param')
# def test_group_matmul_given_head_num_kv_head_num_a_b_is_k_when_is_int8_flag_true_then_correct_shape(mock_get_quant_param):
#     op = OpcheckPagedAttentionAttentionOperation()
#     head_num = 8
#     kv_head_num = 4
#     a = torch.randn(8, 16, 32)
#     b = torch.randn(4, 16, 32)
#     is_k = True
#     mock_get_quant_param.return_value = (True, True, [torch.randn(16), torch.randn(16), torch.randn(16), torch.randn(16)])
#     score = op.group_matmul(head_num, kv_head_num, a, b, is_k)
#     assert score.shape == (8, 16, 32)

# @patch.object(OpcheckPagedAttentionAttentionOperation, 'get_quant_param')
# def test_group_matmul_given_head_num_kv_head_num_a_b_is_k_when_is_int8_flag_false_then_correct_shape(mock_get_quant_param):
#     op = OpcheckPagedAttentionAttentionOperation()
#     head_num = 8
#     kv_head_num = 4
#     a = torch.randn(8, 16, 32)
#     b = torch.randn(4, 16, 32)
#     is_k = True
#     mock_get_quant_param.return_value = (False, False, [None, None, None, None])
#     score = op.group_matmul(head_num, kv_head_num, a, b, is_k)
#     assert score.shape == (8, 16, 32)

# # Test cases for ref_masked_attention
# @patch.object(OpcheckPagedAttentionAttentionOperation, 'group_matmul')
# def test_ref_masked_attention_given_masked_attention_input_alibi_bias_when_quant_type_undefined_then_correct_shape(mock_group_matmul):
#     op = OpcheckPagedAttentionAttentionOperation()
#     masked_attention_input = [torch.randn(1, 8, 16), torch.randn(16, 4, 16), torch.randn(16, 4, 16), 1.0]
#     alibi_bias = None
#     mock_group_matmul.return_value = torch.randn(8, 1, 16)
#     out = op.ref_masked_attention(masked_attention_input, alibi_bias)
#     assert out.shape == (1, 8, 16)

@patch.object(OpcheckPagedAttentionAttentionOperation, 'group_matmul')
def test_ref_masked_attention_given_masked_attention_input_alibi_bias_when_quant_type_dequant_fusion_then_correct_shape(mock_group_matmul):
    op = OpcheckPagedAttentionAttentionOperation()
    masked_attention_input = [torch.randn(1, 8, 16), torch.randn(16, 4, 16), torch.randn(16, 4, 16), 1.0]
    alibi_bias = torch.randn(8, 1, 16)
    mock_group_matmul.return_value = torch.randn(8, 1, 16)
    out = op.ref_masked_attention(masked_attention_input, alibi_bias)
    assert out.shape == (1, 8, 16)

# # Test cases for ref_single_query_cached_kv_attention
# @patch.object(OpcheckPagedAttentionAttentionOperation, 'ref_masked_attention')
# def test_ref_single_query_cached_kv_attention_given_output_paged_input_mask_when_mask_type_undefined_then_correct_shape(mock_ref_masked_attention):
#     op = OpcheckPagedAttentionAttentionOperation()
#     output = torch.randn(4, 8, 16)
#     paged_input = [torch.randn(4, 8, 16), torch.randn(16, 4, 16), torch.randn(16, 4, 16), torch.randn(4, 4), [4, 8, 12, 16]]
#     mask = None
#     mock_ref_masked_attention.return_value = torch.randn(8, 16)
#     op.ref_single_query_cached_kv_attention(output, paged_input, mask)
#     assert output.shape == (4, 8, 16)

# @patch.object(OpcheckPagedAttentionAttentionOperation, 'ref_masked_attention')
# def test_ref_single_query_cached_kv_attention_given_output_paged_input_mask_when_mask_type_norm_then_correct_shape(mock_ref_masked_attention):
#     op = OpcheckPagedAttentionAttentionOperation()
#     output = torch.randn(4, 8, 16)
#     paged_input = [torch.randn(4, 8, 16), torch.randn(16, 4, 16), torch.randn(16, 4, 16), torch.randn(4, 4), [4, 8, 12, 16]]
#     mask = torch.randn(4, 1, 16)
#     mock_ref_masked_attention.return_value = torch.randn(8, 16)
#     op.ref_single_query_cached_kv_attention(output, paged_input, mask)
#     assert output.shape == (4, 8, 16)

# # Test cases for golden_calc
# @patch.object(OpcheckPagedAttentionAttentionOperation, 'ref_single_query_cached_kv_attention')
# def test_golden_calc_given_in_tensors_when_mask_type_undefined_then_correct_shape(mock_ref_single_query_cached_kv_attention):
#     op = OpcheckPagedAttentionAttentionOperation()
#     in_tensors = [torch.randn(4, 8, 16), torch.randn(16, 4, 16), torch.randn(16, 4, 16), torch.randn(4, 4), [4, 8, 12, 16]]
#     mock_ref_single_query_cached_kv_attention.return_value = None
#     ref_output = op.golden_calc(in_tensors)
#     assert ref_output[0].shape == (4, 8, 16)

# @patch.object(OpcheckPagedAttentionAttentionOperation, 'ref_single_query_cached_kv_attention')
# def test_golden_calc_given_in_tensors_when_mask_type_norm_then_correct_shape(mock_ref_single_query_cached_kv_attention):
#     op = OpcheckPagedAttentionAttentionOperation()
#     in_tensors = [torch.randn(4, 8, 16), torch.randn(16, 4, 16), torch.randn(16, 4, 16), torch.randn(4, 4), [4, 8, 12, 16], torch.randn(4, 1, 16)]
#     mock_ref_single_query_cached_kv_attention.return_value = None
#     ref_output = op.golden_calc(in_tensors)
#     assert ref_output[0].shape == (4, 8, 16)