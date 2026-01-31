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

from msit_llm.opcheck import operation_test


class OpcheckPadOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        tmp_out = in_tensors[0]
        padding_offset = in_tensors[1]
        seq_len = in_tensors[2]
        input_ids = in_tensors[3]
        batch = input_ids.shape[0]
        hidden_dim = tmp_out.shape[1]
        max_seq_len = input_ids.shape[1]

        golden_result = torch.zeros((batch, hidden_dim))
        temp_val = 0
        for i in range(batch):
            temp_val = temp_val + seq_len[i][0]
            golden_result[i] = tmp_out[temp_val - 1]
        return [golden_result]

    def test(self):
        self.execute()