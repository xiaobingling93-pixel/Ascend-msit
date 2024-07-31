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
import json
import math
import torch
import torch_npu
import numpy as np

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class CalcType(Enum):
    UNDEFINED = 0 # decoder&encoder for flashAttention
    ENCODER = 1 # encoder for flashAttention
    DECODER = 2 # decoder for flashAttention
    PA_ENCODER = 3 # encoder for pagedAttention


class KvCacheCfg(Enum):
    K_CACHE_V_CACHE = 0 # 默认值，进行kvcache处理
    K_BYPASS_V_BYPASS = 1 # 直接传入kvcache


class MaskType(Enum):
    MASK_TYPE_UNDEFINED = 0
    MASK_TYPE_NROM = 1
    MASK_TYPE_ALIBI = 2
    MASK_TYPE_NORM_COMPRESS = 3
    MASK_TYPE_ALIBI_COMPRESS = 4
    MASK_TYPE_ALIBI_COMPRESS_SQRT = 5
    MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN = 6


class KernelType(Enum):
    KERNELTYPE_DEFAULT = 0
    KERNELTYPE_HIGH_PRECISION = 1


class OpcheckUnpadSelfAttentionOperation(operation_test.OperationTest):
    @staticmethod
    def get_asdops_param(in_tensors, op_param):
        q, k, v, mask, seq_len = in_tensors
        asdops_param = {}
        asdops_param["head_num"] = op_param.get("headNum", 0)
        asdops_param["is_decoder"] = False
        asdops_param["embeddim"] = int(q.shape[1] / asdops_param["head_num"])
        asdops_param["kv_head"] = op_param.get("kvHeadNum", 0)
        asdops_param["is_mask"] = op_param.get("MaskType", MaskType.MASK_TYPE_UNDEFINED) != MaskType.MASK_TYPE_UNDEFINED
        asdops_param["qk_scale"] = op_param.get("qkScale", 1.0)
        asdops_param["post_mask_coff"] = -3e38
        if op_param.get("kernelType", KernelType.KERNELTYPE_DEFAULT) == KernelType.KERNELTYPE_HIGH_PRECISION:
            asdops_param["post_mask_coff"] = 1
        
        asdops_param["data_type"] = q.dtype
        asdops_param["q_ntokens"] = q.shape[0]
        asdops_param["kv_ntokens"] = k.shape[0]
        asdops_param["q_seqlen"] = seq_len.tolist()
        asdops_param["maskType"] = op_param.get("MaskType", MaskType.MASK_TYPE_UNDEFINED)

        MASK_TYPE_NO_MASK = 0
        MASK_TYPE_NO_HEAD = 1
        MASK_TYPE_NO_BATCH = 2
        MASK_TYPE_ALIBI_WITH_BATCH = 3
        MASK_TYPE_ALIBI_NO_BATCH = 4
        MASK_TYPE_NO_HEAD_DECODER = 5

        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH: (( lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH: (( lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD: (( lambda mask, idx, q_s, kv_s: (mask[idx, :q_s:, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER: (( lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH: (( lambda mask, idx, q_s, kv_s: (mask[:, :q_s:, :kv_s]))),
            # 不加mask
            MASK_TYPE_NO_HEAD: (( lambda mask, idx, q_s, kv_s: 0))
        }

        asdops_mask_type = 0
        asdops_param["mask_info"] = mask_type_dict[asdops_mask_type]

        return asdops_param

    def group_matmul(self, heads, group_num, in_a, in_b):
        try:
            group_head = heads // group_num
        except ZeroDivisionError as e:
            raise RuntimeError("get ZeroDivisionError when calc SelfAttentionOperation golden") from e     
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
    
    @staticmethod
    def reshape_qkv(qkv):
        # kv的计算方式似乎与q不同
        if len(qkv.shape) == 3:
            dim0, dim1, dim2 = qkv.shape
            qkv = qkv.contiguous().view(dim0, dim1 * dim2)
        elif len(qkv.shape) == 4:
            dim0, dim1, dim2, dim3 = qkv.shape
            qkv = qkv.contiguous().view(dim0 * dim1, dim2 * dim3)
        return qkv

    def golden_calc(self, in_tensors):
        kvcache_cfg = self.op_param.get("kvcacheCfg", KvCacheCfg.K_CACHE_V_CACHE)
        if kvcache_cfg == KvCacheCfg.K_BYPASS_V_BYPASS:
            q_offset, k_offset, v_offset = 0, 0, 0
            isdecoder = 1
            is_clamp = 0
            batch = 
            heads = 
            embed = 
            max_seq = 
            q_seqlen = in_tensors[5]
            kv_seqlen = in_tensors[4]
            kv_head = 
            mask = in_tensors[3]
            is_mask = True
            q = in_tensors[0]
            k = in_tensors[1]
            v = in_tensors[2]
            q_ntokens = sum(q_seqlen)
            kv_ntokens = sum(kv_seqlen)
            layer_id = in_tensors[6][0]
            is_multi_layer = True
            s = None
            _p = None
            out = None

            for idx in range(batch):
                q_s = q_seqlen[idx]
                kv_s = kv_seqlen[idx]
                q_slice = q[q_offset:q_offset + q_s][:].view(q_s, heads, embed).permute(1, 0, 2)
                k_slice_t = k[layer_id][idx][:kv_s][:].view(kv_s, kv_head, embed).permute(1, 2, 0) # get K^T
                v_slice = v[layer_id][idx][:kv_s][:].view(kv_s, kv_head, embed).permute(1, 0, 2)

                score = self.group_matmul(heads, kv_head, q_slice, k_slice_t)

                if s is None:
                    s = score.view([-1, ])
                else:
                    s = torch.concat((s, score.view([-1, ])), 0)

                score = 1
                if not is_multi_layer:
                    # 当前scale和tor保持一致，模型侧可能传入scale = np.float32(layer_id + 1)
                    scale = layer_id + 1
                score = score * scale
                
                if is_clamp == 1:
                    clamp_min_brc = torch.ones(score.shape) * clamp_min
                    clamp_max_brc = torch.ones(score.shape) * clamp_max
                    score = torch.max(score, clamp_min_brc)
                    score = torch.min(score, clamp_max_brc)
                if is_mask:
                    score = score + mask[idx, :q_s, :kv_s]

                score_max = torch.max(score, axis=-1)
                score = score - score_max.reshape((heads, q_s, 1))
                score_exp = torch.exp(score)
                score_sum = torch.sum(score_exp, axis=-1)

                if _p is None:
                    _p = score_exp.reshape([-1, ])
                else:
                    _p = torch.concat((_p, score_exp.reshape([-1, ])), 0)

                p = (score_exp / score_sum.reshape((heads, q_s, 1)))
                o = self.group_matmul(heads, kv_head, p, v_slice)
                o = o.view(heads, q_s, embed).permute(1, 0, 2).contiguous()
                if out is None:
                    out = o
                else:
                    out = torch.concat((out, o), 0)
                
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
            
            # golden data
            out = out.view(q_ntokens, heads * embed)
            golden_out = out.to(q.dtype)
            return [golden_out]

        calc_type = self.op_param.get("calcType", CalcType.UNDEFINED)
        logger_text = f"CalcType: {calc_type}"
        logger.debug(logger_text)

        if calc_type == CalcType.PA_ENCODER:
            q = in_tensors[0]
            k = in_tensors[1]
            v = in_tensors[2]
            q = OpcheckUnpadSelfAttentionOperation.reshape_qkv(q)
            k = OpcheckUnpadSelfAttentionOperation.reshape_qkv(k)
            v = OpcheckUnpadSelfAttentionOperation.reshape_qkv(v)
            mask = None
            seq_len = None
            mask_type = self.op_param.get("maskType", MaskType.MASK_TYPE_UNDEFINED)
            if mask_type != MaskType.MASK_TYPE_UNDEFINED:
                mask = in_tensors[3]
                seq_len = in_tensors[4]
            else:
                seq_len = in_tensors[3]
            if mask is not None:
                if self.get_soc_version() != "Ascend910B":
                    mask = mask.contiguous().permute(0, 2, 1, 3)
                    dim0, dim1, dim2, dim3 = mask.shape
                    mask = mask.contiguous().view(dim0, dim1, dim2 * dim3)
                    mask = mask[:, :, : dim1]
                    batch = len(seq_len)
                    if dim0 == 1:
                        mask = mask.contiguous().view(dim1, dim1)
                    elif dim1 != batch:
                        if dim1 > dim2 * dim3:
                            # alibi mask compress
                            step = dim2 * dim3
                            mask_padded = -10000 * torch.ones(dim0, dim1 * 2, dim1)
                            for i in range(dim1 // step):
                                dim1_offset = i * step
                                dim2_offset = i * step
                                mask_padded[:, dim1_offset : dim1_offset + dim1, dim2_offset : dim2_offset + step] = mask
                            mask = mask_padded[:, :dim1, :dim1]
            in_tensors = [q, k, v, mask, seq_len]
            asdops_params = OpcheckUnpadSelfAttentionOperation.get_asdops_param(in_tensors, self.op_param)
        elif calc_type == CalcType.ENCODER:
            pass
        else:
            pass
            
        return [out]

    def test(self):
        self.execute()