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


class OpcheckElewiseSubOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        out = []
        seq_len = self.op_param.get("seqLen", None)
        head_num = self.op_param.get("headNum", None)
        for i, s in enumerate(seq_len):
            for _ in range(head_num):
                out.append(in_tensors[0][i, :, :s, :s].flatten())
        return [torch.hstack(out)]

    def test_2d_half(self):
        ret = self.validate_param("seqLen", "headNum")
        if not ret:
            return
        self.execute()