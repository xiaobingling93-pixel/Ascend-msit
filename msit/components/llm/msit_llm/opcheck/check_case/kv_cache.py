# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import torch
import torch_npu

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckKvCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        # cache_in与cache_out只支持float16/int8数据类型
        newkv = in_tensors[0]
        layer_id = in_tensors[1]
        cache_in = in_tensors[2]
        token_offset = in_tensors[3]
        seqlen = in_tensors[4]
        cache_out = torch.zeros(shape=cache_in.shape).type(cache_in.dtype)
        batch = len(seqlen)

        prefix_ntokens = 0
        for i in range(batch):
            for j in range(seqlen[i]):
                cache_out[layer_id[0]][i][token_offset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
            prefix_ntokens += seqlen[i]

        return [newkv, layer_id, cache_out, token_offset, seqlen]

    def test(self):
        self.execute_inplace()