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


class OpcheckCumsumOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        axes = self.op_param.get("axes", None)
        exclusive = self.op_param.get("exclusive", False) # 暂不支持true
        reverse = self.op_param.get("reverse", False) # 暂不支持true
        golden_result = torch.cumsum(in_tensors[0], dim=axes[0])
        return [golden_result]

    def test(self):
        ret = self.validate_param("axes")
        if not ret:
            return
        self.execute()