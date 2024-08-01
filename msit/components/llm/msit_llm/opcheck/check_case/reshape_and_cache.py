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
    COMPRESS_TYPE_KVHEAD = 1 # 压缩key_cache, value_cache的kvHead维度


class OpcheckReshapeAndCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        num_tokens, num_heads, head_size = in_tensors[0].shape
        data_type = in_tensors[0].dtype
        slot_mapping = in_tensors[4]
        soc_version = self.get_soc_version()
        if soc_version == "Ascend910B":
            num_blocks, block_size, _, _ = in_tensors[2].shape
            key_expect = in_tensors[2]
            value_expect = in_tensors[3]
            compress_type = self.op_param.get("compressType", CompressType.COMPRESS_TYPE_UNDEFINED.value)
            if compress_type == CompressType.COMPRESS_TYPE_KVHEAD.value:
                wins = in_tensors[5]
                seq_len = in_tensors[6]
                new_seq = seq_len
                new_seq[0] = seq_len[0]
                for n in range(1, len(seq_len)):
                    new_seq[n] = seq_len[n] + seq_len[n - 1]

                for i, slot in enumerate(slot_mapping):
                    if slot < 0:
                        continue
                    cur_slot = slot
                    win = wins[i]
                    for j in range(win):
                        block_index = cur_slot // block_size
                        block_offset = cur_slot % block_size
                        cur_batch = i // num_heads
                        bs_id = new_seq[cur_batch] - win + j
                        head_id = i % num_heads
                        token_key = in_tensors[0][bs_id][head_id]
                        toekn_v =  in_tensors[1][bs_id][head_id]
                        key_expect[block_index][block_offset] = token_key
                        value_expect[block_index][block_offset] = toekn_v
                        cur_slot += 1
                return [key_expect, value_expect]
            else:
                for i, slot in enumerate(slot_mapping):
                    if slot < 0:
                        continue
                    block_index = slot // block_size
                    block_offset = slot % block_size

                    token_key = in_tensors[0][i]
                    token_v = in_tensors[1][i]

                    key_expect[block_index][block_offset] = token_key
                    value_expect[block_index][block_offset] = token_v
                return [key_expect, value_expect]
        else:
            num_blocks, _, block_size, _ = in_tensors[2].shape # key_cache
            key_expect_nz = in_tensors[2]
            value_expect_nz = in_tensors[3]
            for i, slot in enumerate(slot_mapping):
                block_index = slot // block_size
                block_offset = slot % block_size
                
                token_key = in_tensors[0][i]
                token_v = in_tensors[1][i]
                token_key = token_key.reshape(num_heads * head_size)
                token_v = token_v.reshape(num_heads * head_size)
                for k in range(num_heads * head_size // 16):
                    key_expect_nz[block_index][k][block_offset][:] = token_key[k * 16 : k * 16 + 16]
                    value_expect_nz[block_index][k][block_offset][:] = token_v[k * 16 : k * 16 + 16]
            return [key_expect_nz, value_expect_nz]

    def test(self):
        ret = self.validate_param("inplace_idx")
        if not ret:
            return
        self.execute_inplace()