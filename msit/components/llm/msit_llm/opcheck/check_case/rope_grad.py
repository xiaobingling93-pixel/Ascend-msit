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
import torch.nn as nn

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckRopeGradOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        # x,128*32-->reshape x,32,128
        qseqlen = self.op_param.get('qSeqLen', None)
        try:
            cos_list = [in_tensors[2][:x, :] for x in qseqlen]
            sin_list = [in_tensors[3][:x, :] for x in qseqlen]
        except Exception as e:
            raise IndexError(f"qSeqLen does not match the size of cos/sin tensors.") from e
        cos = torch.concat(cos_list, dim=0)
        sin = torch.concat(sin_list, dim=0)
        sin1 = sin[:, :64]
        sin2 = sin[:, 64:]
        rohqgsin = torch.concat((sin2, -sin1), dim=-1)
        q_grad = torch.zeros_like(in_tensors[0])
        bs = int(in_tensors[0].shape[1] / 128)
        for i in range(bs):
            q_grad[:, i * 128:(i + 1) * 128] = in_tensors[0][:, i * 128:(i + 1) * 128] * (cos + rohqgsin)

        k_grad = torch.zeros_like(in_tensors[1])
        for i in range(bs):
            k_grad[:, i * 128:(i + 1) * 128] = in_tensors[1][:, i * 128:(i + 1) * 128] * (cos + rohqgsin)
        return [q_grad, k_grad]

    def test(self):
        ret = self.validate_param("qSeqLen")
        if not ret:
            return
        self.execute()