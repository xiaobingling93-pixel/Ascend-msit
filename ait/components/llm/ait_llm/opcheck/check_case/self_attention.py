# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import json
import math
import torch
import torch_npu
import numpy as np

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckUnpadSelfAttentionOperation(operation_test.OperationTest):
    def group_matmul(self, heads, group_num, in_a, in_b):
        try:
            group_head = heads // group_num
        except ZeroDivisionError as e:
            raise e       
        score = None
        for i in range(group_num):
            group_score = torch.matmul(in_a[i * group_head: (i + 1) * group_head, :, :], in_b[i:(i + 1), :, :])
            if score is None:
                score = group_score
            else:
                score = torch.concat((score, group_score), 0)
        logger.debug(score.shape)
        return score

    def undefined_golden_func(self, in_tensors):
        mixed_q, mixed_k, mixed_v, cache_k, cache_v, attention_mask, token_offset, seq_len, layerid = in_tensors[0], \
            in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6], in_tensors[7], \
            int(in_tensors[8][0])

        soc_version = self.get_soc_version()
        if soc_version == 'Ascend310P':
            cache_k = self.nz_2_nd(cache_k)
            cache_v = self.nz_2_nd(cache_v)
            attention_mask = self.nz_2_nd(attention_mask)

        batch_run_status_enable = self.op_param.get("batchRunStatusEnable", False)
        if batch_run_status_enable:
            batch_status = in_tensors[9]
        else:
            batch_status = len(seq_len)
        q_scale, qk_scale, head_num, head_size = self.op_param.get("qScale", 1.0), self.op_param.get("qkScale", 1.0), \
            self.op_param.get("headNum", 32), mixed_k.size(-1)
        
        offset = 0
        context_list = []

        for i in range(batch_status):
            cur_seqlen = seq_len[i]
            cur_token_offset = token_offset[i]
            cur_token_offset_start = cur_token_offset - cur_seqlen
            next_offset = offset + cur_seqlen
            cur_q = mixed_q[offset:next_offset]
            cur_k = mixed_k[offset:next_offset]
            cur_v = mixed_v[offset:next_offset]
            if cur_token_offset_start > 0:
                past_k = cache_k[layerid, i, :cur_token_offset_start, :]
                past_v = cache_v[layerid, i, :cur_token_offset_start, :]
                cur_k = torch.concat([past_k, cur_k], dim=0)
                cur_v = torch.concat([past_v, cur_v], dim=0)
            cur_q = (cur_q * q_scale).view(cur_seqlen, head_num, head_size).transpose(0, 1)
            cur_k = cur_k.view(cur_token_offset, head_num, head_size).permute(1, 2, 0)
            cur_qk = torch.bmm(cur_q, cur_k) # [head_num, seqlen, token_offset]
            if self.op_param.get("clampType", 0):
                clamp_min = self.op_param.get("clampMin", 0.0)
                clamp_max = self.op_param.get("clampMax", 0.0)
                cur_qk = torch.clamp(cur_qk, clamp_min, clamp_max)
            if attention_mask.ndim == 3: # masked_fill
                cur_qk = cur_qk + attention_mask[i, :cur_seqlen, :cur_token_offset]
            else:
                cur_qk = cur_qk + attention_mask[:cur_seqlen, :cur_token_offset]
            cur_qk = cur_qk * qk_scale
            cur_qk = torch.nn.functional.softmax(cur_qk, dim=-1)
            cur_v = cur_v.view(cur_token_offset, head_num, head_size).transpose(0, 1)
            cur_context = torch.bmm(cur_qk, cur_v).transpose(0, 1).contiguous().view(cur_seqlen, head_num * head_size)
            context_list.append(cur_context)

            offset = next_offset

        out = torch.concat(context_list, dim=0)
        return out

    def encoder_golden_func(self, in_tensors):
        mixed_q, mixed_k, mixed_v, attention_mask, seq_len = in_tensors[0], in_tensors[1], in_tensors[2], \
            in_tensors[3], in_tensors[4]
        
        soc_version = self.get_soc_version()
        if soc_version == 'Ascend310P':
            attention_mask = self.nz_2_nd(attention_mask)

        dtype = mixed_q.dtype
        batch_run_status_enable = self.op_param.get("batchRunStatusEnable", False)
        if batch_run_status_enable:
            batch_status = in_tensors[5]
        else:
            batch_status = len(seq_len)

        heads, group_num, embed = self.op_param.get("headNum", 32), self.op_param.get("kvHeadNum", 32), mixed_k.size(-1)
        q_seqlen = kv_seqlen = seq_len # crossattention时，q_seqlen != k_seqlen

        q_offset, k_offset, v_offset = 0, 0, 0
        s, _p, out = None, None, None

        for idx in range(batch_status):
            q_s, kv_s = q_seqlen[idx], kv_seqlen[idx]
            q_slice = mixed_q[q_offset:q_offset + q_s][:].view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
            k_slice = mixed_k[k_offset:k_offset + kv_s][:].view(kv_s, group_num, embed)
            k_slice = torch.permute(k_slice, (1, 0, 2))
            k_slice_t = torch.permute(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
            v_slice = mixed_v[v_offset:v_offset + kv_s][:].view(kv_s, group_num, embed)
            v_slice = torch.permute(v_slice, (1, 0, 2))
            score = self.group_matmul(heads, group_num, q_slice, k_slice_t)
            s = score.view(-1) if s is None else torch.concat((s, score.view(-1)), 0)

            score = score / math.sqrt(1.0 * embed)
            score_max = torch.max(score, axis=-1).values
            score = score - score_max.view((heads, q_s, 1))
            score_exp = torch.exp(score)
            if not dtype == torch.float32:
                score_sum = torch.sum(score_exp, axis=-1)
                _p = score_exp.view(-1) if _p is None else torch.concat((_p, score_exp.view(-1)), 0)
                p = score_exp / score_sum.view(heads, q_s, 1)
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
            else:
                score_sum = torch.sum(score_exp, axis=-1)
                _p = score_exp.view(-1) if _p is None else torch.concat((_p, score_exp.view(-1)), 0)
                p = score_exp
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
                out_sub = out_sub / score_sum.view(heads, q_s, 1)
            out_sub = out_sub.view(heads, q_s, embed)
            out_sub = torch.permute(out_sub, (1, 0, 2))
            out = out_sub if out is None else torch.concat((out, out_sub), 0)

        return out.view(-1, heads, embed)

    def decoder_golden_func(self, in_tensors):
        return self.undefined_golden_func(in_tensors)

    def pa_encoder_golden_func(self, in_tensors):
        return self.encoder_golden_func(in_tensors)

    def golden_calc(self, in_tensors):
        calc_type = self.op_param.get("calcType", 0)
        if calc_type == 0:
            logger_text = f"CalcType: {calc_type}. Undefined calcType!"
            logger.debug(logger_text)
            out = self.undefined_golden_func(in_tensors)
        elif calc_type == 1:
            logger_text = f"CalcType: {calc_type}. Using encoder of FlashAttention!"
            logger.debug(logger_text)
            out = self.encoder_golden_func(in_tensors)
        elif calc_type == 2:
            logger_text = f"CalcType: {calc_type}. Using decoder of FlashAttention!"
            logger.debug(logger_text)
            out = self.decoder_golden_func(in_tensors)
        else:
            logger_text = f"CalcType: {calc_type}. Using encoder of PageAttention!"
            logger.debug(logger_text)
            out = self.pa_encoder_golden_func(in_tensors)
        return [out]

    def test(self):
        self.execute()