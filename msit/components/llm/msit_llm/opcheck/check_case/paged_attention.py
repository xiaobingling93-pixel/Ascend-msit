# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import torch
import torch_npu

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class CompressType(Enum):
    COMPRESS_TYPE_UNDEFINED = 0 # 默认值，不压缩
    COMPRESS_TYPE_KVHEAD = 1 # 压缩key_cache, value_cache的kvHead维度，只支持Atlas 800I A2


class MaskType(Enum):
    UNDEFINED = 0 # 默认值，全0的mask
    MASK_TYPE_NONE = 1 # 倒三角mask
    MASK_TYPE_ALIBI = 2 # alibi mask
    MASK_TYPE_SPEC = 3 # 并行解码mask


class QuantType(Enum):
    TYPE_QUANT_UNDEFINED = 0
    TYPE_DEQUANT_FUSION = 1


class OpcheckPagedAttentionAttentionOperation(operation_test.OperationTest):
    @staticmethod
    def get_asdops_param(in_tensors, op_param):
        asdops_param = {}
        asdops_param['num_heads'] = op_param.get('headNum', 0)
        asdops_param['qkScale'] = op_param.get('qkScale', 1.0)
        asdops_param['num_tokens'], _, asdops_param['head_size'] = in_tensors[0].shape # q shape
        asdops_param['kv_heads'] = op_param.get('kvHeadNum', 0)
        if op_param['kvHeadNum'] == 0:
            asdops_param['kv_heads'] = asdops_param['num_heads']
        if op_param['compressType'] == CompressType.COMPRESS_TYPE_KVHEAD.value:
            asdops_param['compressHead'] = True
        else:
            asdops_param['compressHead'] = False
        context_lens = in_tensors[4]
        asdops_param['max_context_len'] = max(context_lens)
        if op_param['maskType'] != MaskType.UNDEFINED.value:
            asdops_param['mask_dim'] = len(in_tensors[5].shape) # mask shape
            asdops_param['mask_data_type'] = in_tensors[5].dtype
            if op_param['maskType'] == MaskType.MASK_TYPE_NONE.value:
                asdops_param['mask_dim'] = 3
        else:
            asdops_param['mask_dim'] = 0
            asdops_param['mask_data_type'] = torch.bfloat16

        if op_param['quantType'] != QuantType.TYPE_QUANT_UNDEFINED.value:
            asdops_param['is_int8_flag'] = True
            if op_param['hasQuantOffset'] == True:
                asdops_param['has_bias'] = True
                asdops_param['offset1'] = in_tensors[8]
                asdops_param['de_scale1_fp32'] = in_tensors[7]
                asdops_param['offset2'] = in_tensors[10]
                asdops_param['de_scale2_fp32'] = in_tensors[9]
            else:
                asdops_param['has_bias'] = False
                asdops_param['de_scale1_fp32'] = in_tensors[7]
                asdops_param['de_scale2_fp32'] = in_tensors[8]
                asdops_param['offset1'] = None
                asdops_param['offset2'] = None
        else:
            asdops_param['is_int8_flag'] = False
            asdops_param['has_bias'] = False
            asdops_param['de_scale1_fp32'] = None
            asdops_param['de_scale2_fp32'] = None
            asdops_param['offset1'] = None
            asdops_param['offset2'] = None
        
        return asdops_param

    @staticmethod
    def group_matmul(asdops_param, head, group_num, A, B, is_k):
        is_int8_flag = asdops_param['is_int8_flag']
        has_bias = asdops_param['has_bias']
        offset1 = asdops_param['offset1']
        de_scale1_fp32 = asdops_param['de_scale1_fp32']
        offset2 = asdops_param['offset2']
        de_scale2_fp32 = asdops_param['de_scale2_fp32']

        group_head = head // group_num
        score = None
        for i in range(group_head):
            if is_int8_flag:
                int8_B = B[i:, (i+1), :, :, ]
                head_dim = int8_B.shape[2]
                int32_B = torch.matmul(torch.eye(int8_B.shape[1].type(torch.int32)), int8_B.type(torch.int32)).type(torch.int32)
                if is_k:
                    if has_bias:
                        int32_B = int32_B + offset1[i * head_dim : (i + 1) * head_dim]
                    fp32_B = int32_B.type(torch.float32) * de_scale1_fp32[i * head_dim : (i + 1) * head_dim]
                    fp32_B = torch.permute(fp32_B, (0, 2, 1))
                else:
                    if has_bias:
                        int32_B = int32_B + offset2[i * head_dim : (i + 1) * head_dim]
                    fp32_B = int32_B.type(torch.float32) * de_scale2_fp32[i * head_dim : (i + 1) * head_dim]
                group_score = torch.matmul(A[i * group_num : (i + 1) * group_num, :, :].type(torch.float32), fp32_B).type(torch.float16)
            else:
                group_score = torch.matmul(A[i * group_num : (i + 1) * group_num, :, :].type(torch.float32), B[i: (i + 1), :, :].type(torch.float32))

            if score is None:
                score = group_score
            else:
                score = torch.concat((score, group_score), 0)
        logger_text = f"score shape: {score.shape}"
        logger.debug(logger_text)
        return score

    @staticmethod
    def ref_masked_attention(asdops_param, masked_attention_input, alibi_bias=None):
        query = masked_attention_input[0] # (1, num_heads, head_size)
        key = masked_attention_input[1] # (context_len, kv_heads, head_size)
        value = masked_attention_input[2]
        scale = masked_attention_input[3] # float

        is_int8_flag = asdops_param['is_int8_flag']
        mask_data_type = asdops_param['mask_data_type']

        # Q * K.T
        query = query * scale
        query = torch.permute(query, (1, 0, 2))
        if not is_int8_flag:
            key = torch.permute(key, (1, 2, 0)) # 0 1 2
        else:
            key = torch.permute(key, (1, 0, 2))
        sim = OpcheckPagedAttentionAttentionOperation.group_matmul(asdops_param, query.shape[0], key.shape[0], query, key, 1) # (head_num, q_seqlen, k_seqlen)
        if alibi_bias is None:
            sim = sim * scale
        else:
            sim = sim * scale
            sim = sim + alibi_bias
        # softmax
        row_max = torch.max(sim, axis=-1, keepdims=True).values
        sim -= row_max
        sim = torch.exp(sim)
        row_sum = torch.sum(sim, axis=-1, keepdims=True)
        p = sim / row_sum
        # P * V
        value = torch.permute(value, (1, 0, 2))
        out = OpcheckPagedAttentionAttentionOperation.group_matmul(asdops_param, query.shape[0], key.shape[0], p, value, 0)
        out = torch.permute(out, (1, 0, 2))
        return out

    @staticmethod
    def ref_single_query_cached_kv_attention(asdops_param, output, paged_input, mask) -> None:
        query = paged_input[0]
        key_cache = paged_input[1] # (num_blocks, block_size, kv_heads, head_size)
        value_cache = paged_input[2] # (num_blocks, block_size, kv_heads, head_size)
        block_tables = paged_input[3]
        context_lens = paged_input[4]

        num_tokens = asdops_param["num_tokens"]
        kv_heads = asdops_param["kv_heads"]
        num_heads = asdops_param["num_heads"]
        head_size = asdops_param["head_size"]
        compress_head = asdops_param["compressHead"]
        max_context_len = asdops_param["max_context_len"]
        mask_dim = asdops_param["mask_dim"]
        mask_data_dim = asdops_param["mask_data_type"]
        mask_index_coff = 1
        if compress_head:
            query = query.view(num_tokens * kv_heads, num_heads // kv_heads, head_size)
            output = output.view(num_tokens * kv_heads, num_heads // kv_heads, head_size)
            if mask_dim == 4:
                mask_shape = mask.shape
                mask = mask.view(mask_shape[0] * kv_heads, num_heads // kv_heads, 1, max_context_len)
            else:
                mask_index_coff = kv_heads
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size = value_cache.shape[3]
        block_size = value_cache.shape[1]

        num_input_tokens = query.shape[0]
        index = 0
        for i in range(len(context_lens)):
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            if context_len == 0:
                continue

            q = query[index].view(1, num_heads, head_size)
            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size)
                keys.append(k)
                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size)
                values.append(v)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            scale = asdops_param['qkScale']
            masked_attention_input = [q, keys, values, scale]
            if mask_dim == 4:
                out = OpcheckPagedAttentionAttentionOperation.ref_masked_attention(asdops_param, masked_attention_input, mask[i, :, :1, :context_len])
                out = out.reshape(num_heads, head_size)
            elif mask_dim == 3:
                out = OpcheckPagedAttentionAttentionOperation.ref_masked_attention(asdops_param, masked_attention_input, mask[i // mask_index_coff, :1, :context_len])
                out = out.reshape(num_heads, head_size)
            else:
                out = OpcheckPagedAttentionAttentionOperation.ref_masked_attention(asdops_param, masked_attention_input, mask)
                out = out.reshape(num_heads, head_size)

            output[index] = out
            index += 1

    @staticmethod
    def cache_nz_2_nd(cache, kv_heads, embedding_size):
        cache = cache.permute(0, 2, 1, 3)
        dim0, dim1, dim2, dim3 = cache.shape
        cache = cache.contiguous().view(dim0, dim1, dim2 * dim3)
        cache = cache[:, :, :embedding_size * kv_heads]
        cache = cache.contiguous().view(dim0, dim1, kv_heads, embedding_size)
        return cache

    def golden_calc(self, in_tensors):
        asdops_param = OpcheckPagedAttentionAttentionOperation.get_asdops_param(in_tensors, self.op_param)
        query, key_cache, value_cache, block_tables, context_lens = in_tensors[:5]
        mask = None

        mask_type = self.op_param.get('maskType', MaskType.UNDEFINED.value)
        if mask_type != MaskType.UNDEFINED.value:
            mask = in_tensors[5]

        soc_version = self.get_soc_version()
        if soc_version == 'Ascend310P':
            kv_heads = self.op_param.get('kvHeads', 1)
            embedding_size = query.shape[-1]

            key_cache = OpcheckPagedAttentionAttentionOperation.cache_nz_2_nd(key_cache, kv_heads, embedding_size)
            value_cache = OpcheckPagedAttentionAttentionOperation.cache_nz_2_nd(value_cache, kv_heads, embedding_size)

            if mask_type != MaskType.UNDEFINED.value:
                # mask nz to nd
                mask = mask.permute(0, 2, 1, 3)
                dim0, dim1, dim2, dim3 = mask.shape
                mask = mask.contiguous().view(dim0, dim1, dim2 * dim3)
                if mask_type == MaskType.MASK_TYPE_ALIBI.value:
                    batch = len(context_lens)
                    head_num = self.op_param.get('headNum', 0)
                    if dim0 != head_num:
                        mask = mask.contiguous().view(batch, head_num, dim1, dim2 * dim3)
                    else:
                        mask = mask.contiguous().view(1, head_num, dim1, dim2 * dim3)

        ref_output = torch.zeros_like(query)
        paged_input = [query, key_cache, value_cache, block_tables, context_lens]
        OpcheckPagedAttentionAttentionOperation.ref_single_query_cached_kv_attention(asdops_param, ref_output, paged_input, mask)
        return [ref_output]

    def test(self):
        self.execute()