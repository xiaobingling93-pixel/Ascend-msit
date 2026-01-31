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


class OpcheckFillOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        with_mask = self.op_param.get("withMask", True)
        out_dim = self.op_param.get("outDim", None)
        value = self.op_param.get("value", None) 

        if with_mask:
            golden_result = in_tensors[0].masked_fill_(in_tensors[1].bool(), value[0])
        else:
            golden_result = torch.full(out_dim, value[0], dtype=torch.float16)
        return [golden_result]

    def test(self):
        ret = self.validate_param("withMask", "outDim", "value")
        if not ret:
            return
        self.execute()