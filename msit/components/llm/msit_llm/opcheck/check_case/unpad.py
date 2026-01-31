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


class OpcheckUnpadOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        input_ids = in_tensors[0]
        cum_offsets_now = in_tensors[1].reshape(-1)
        token_num = in_tensors[2]
        seq_len = in_tensors[3]
        batch = in_tensors[0].shape[0]
        total_length_imm = in_tensors[0].shape[1]

        x_remove_padding = input_ids[0][0:seq_len[0]]
        for i in range(1, batch):
            x_remove_padding = torch.concat((x_remove_padding, input_ids[i][0:seq_len[i]]))
        x_remove_padding = torch.pad(x_remove_padding, (0, (batch * total_length_imm - token_num[0][0])))
        cum_offsets_out = torch.zeros(batch)
        for i in range(1, batch):
            cum_offsets_out[i] = cum_offsets_now[i - 1]
        padding_offset = seq_len[0] * [0]
        for i in range(1, batch):
            temp_pad_out = seq_len[i] * [cum_offsets_now[i - 1]]
            padding_offset = torch.concat((padding_offset, temp_pad_out))
        zero_offset = torch.zeros((1, batch * total_length_imm - token_num[0][0]))
        padding_offset = torch.append(padding_offset, zero_offset)
        x_remove_padding = x_remove_padding.reshape(1, batch * total_length_imm).long()
        cum_offsets_out = cum_offsets_out.reshape(batch, 1).int()
        padding_offset = padding_offset.reshape(1, batch * total_length_imm).int()
        return [x_remove_padding, cum_offsets_out, padding_offset] # 输出 int64, int32, int32

    def test(self): 
        self.execute()