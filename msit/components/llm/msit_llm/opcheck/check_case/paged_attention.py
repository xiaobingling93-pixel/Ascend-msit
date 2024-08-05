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
    MASK_TYPE_NORM = 1 # 倒三角mask
    MASK_TYPE_ALIBI = 2 # alibi mask
    MASK_TYPE_SPEC = 3 # 并行解码mask


class QuantType(Enum):
    TYPE_QUANT_UNDEFINED = 0
    TYPE_DEQUANT_FUSION = 1


class OpcheckPagedAttentionAttentionOperation(operation_test.OperationTest):
    def group_matmul(self, head_num, kv_head_num, A, B, is_k):
        quant_type = self.op_param.get('quantType', QuantType.TYPE_QUANT_UNDEFINED.value)
        has_quant_offset = self.op_param.get('hasQuantOffset', False)
        if quant_type!= QuantType.TYPE_QUANT_UNDEFINED.value:
            is_int8_flag = True
            if has_quant_offset:
                has_bias = True
                offset1 = self.in_tensors[8]
                de_scale1_fp32 = self.in_tensors[7]
                offset2 = self.in_tensors[10]
                de_scale2_fp32 = self.in_tensors[9]
            else:
                has_bias = False
                de_scale1_fp32 = self.in_tensors[7]
                de_scale2_fp32 = self.in_tensors[8]
                offset1 = None
                offset2 = None
        else:
            is_int8_flag = False
            has_bias = False
            de_scale1_fp32 = None
            de_scale2_fp32 = None
            offset1 = None
            offset2 = None

        group_head = head_num // kv_head_num
        score = None
        for i in range(kv_head_num):
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
                group_score = torch.matmul(A[i * group_head : (i + 1) * group_head, :, :].type(torch.float32), fp32_B).type(torch.float16)
            else:
                group_score = torch.matmul(A[i * group_head : (i + 1) * group_head, :, :].type(torch.float32), B[i: (i + 1), :, :].type(torch.float32))

            if score is None:
                score = group_score
            else:
                score = torch.concat((score, group_score), 0)
        logger_text = f"score shape: {score.shape}"
        logger.debug(logger_text)
        return score

    def ref_masked_attention(self, masked_attention_input, alibi_bias=None):
        query = masked_attention_input[0] # (1, head_num, head_size)
        key = masked_attention_input[1] # (context_len, kv_head_num, head_size)
        value = masked_attention_input[2]
        scale = masked_attention_input[3] # float

        # Q * K.T
        query = torch.permute(query, (1, 0, 2))
        quant_type = self.op_param.get('quantType', QuantType.TYPE_QUANT_UNDEFINED.value)
        if quant_type == QuantType.TYPE_QUANT_UNDEFINED.value:
            key = torch.permute(key, (1, 2, 0)) # 0 1 2
        else:
            key = torch.permute(key, (1, 0, 2))
        sim = self.group_matmul(query.shape[0], key.shape[0], query, key, 1) # (head_num, q_seqlen, k_seqlen)
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
        out = self.group_matmul(query.shape[0], key.shape[0], p, value, 0)
        out = torch.permute(out, (1, 0, 2))
        return out

    def ref_single_query_cached_kv_attention(self, output, paged_input, mask) -> None:
        query = paged_input[0]
        key_cache = paged_input[1] # (num_blocks, block_size, kv_head_num, head_size)
        value_cache = paged_input[2] # (num_blocks, block_size, kv_head_num, head_size)
        block_tables = paged_input[3]
        context_lens = paged_input[4]

        head_num = self.op_param.get('headNum', 0)
        kv_head_num = self.op_param.get('kvHeadNum', 0)
        num_tokens, _, head_size = query.shape
        _, block_size, _, _ = key_cache.shape
        max_context_len = max(context_lens)

        mask_type = self.op_param.get('maskType', MaskType.UNDEFINED.value)
        if mask_type != MaskType.UNDEFINED.value:
            mask_dim = len(mask) # mask shape
            if mask_type == MaskType.MASK_TYPE_NORM.value:
                mask_dim = 3
        else:
            mask_dim = 0
        mask_index_coff = 1

        compress_type = self.op_param.get('compressType', CompressType.COMPRESS_TYPE_UNDEFINED.value)
        if compress_type == CompressType.COMPRESS_TYPE_KVHEAD.value:
            query = query.view(num_tokens * kv_head_num, head_num // kv_head_num, head_size)
            output = output.view(num_tokens * kv_head_num, head_num // kv_head_num, head_size)
            if mask_dim == 4:
                mask_shape = mask.shape
                mask = mask.view(mask_shape[0] * kv_head_num, head_num // kv_head_num, 1, max_context_len)
            else:
                mask_index_coff = kv_head_num

        for i in range(num_tokens):
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            if context_len == 0:
                continue

            q = query[i].view(1, head_num, head_size)
            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_head_num, head_size)
                keys.append(k)
                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_head_num, head_size)
                values.append(v)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            scale = self.op_param.get('qkScale', 1.0)
            masked_attention_input = [q, keys, values, scale]
            if mask_dim == 4:
                out = self.ref_masked_attention(masked_attention_input, mask[i, :, :1, :context_len])
                out = out.reshape(head_num, head_size)
            elif mask_dim == 3:
                out = self.ref_masked_attention(masked_attention_input, mask[i // mask_index_coff, :1, :context_len])
                out = out.reshape(head_num, head_size)
            else:
                out = self.ref_masked_attention(masked_attention_input, mask)
                out = out.reshape(head_num, head_size)

            output[i] = out

    def golden_calc(self, in_tensors):
        query, key_cache, value_cache, block_tables, context_lens = in_tensors[:5]
        mask = None

        head_num = self.op_param.get('headNum', 0)
        kv_head_num = self.op_param.get('kvHeadNum', 0)
        num_tokens, _, head_size = query.shape

        mask_type = self.op_param.get('maskType', MaskType.UNDEFINED.value)
        if mask_type != MaskType.UNDEFINED.value:
            mask = in_tensors[5]

        soc_version = self.get_soc_version()
        if soc_version == 'Ascend310P':
            num_blocks, _, block_size, _ = key_cache.shape
            key_cache = key_cache.permute(0, 2, 1, 3).reshape(num_blocks, block_size, kv_head_num, head_size)
            value_cache = value_cache.permute(0, 2, 1, 3).reshape(num_blocks, block_size, kv_head_num, head_size)

            if mask_type != MaskType.UNDEFINED.value:
                # mask nz to nd
                mask = mask.permute(0, 2, 1, 3)
                dim0, dim1, dim2, dim3 = mask.shape
                mask = mask.contiguous().view(dim0, dim1, dim2 * dim3)
                if mask_type == MaskType.MASK_TYPE_ALIBI.value:
                    batch = len(context_lens)
                    if dim0 != head_num:
                        mask = mask.contiguous().view(batch, head_num, dim1, dim2 * dim3)
                    else:
                        mask = mask.contiguous().view(1, head_num, dim1, dim2 * dim3)

        ref_output = torch.zeros_like(query)
        paged_input = [query, key_cache, value_cache, block_tables, context_lens]
        self.ref_single_query_cached_kv_attention(ref_output, paged_input, mask)
        return [ref_output]

    def test(self):
        self.execute()