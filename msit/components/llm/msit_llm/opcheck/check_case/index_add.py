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
from msit_llm.opcheck import operation_test


class OpcheckIndexAddOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        index_type = self.op_param.get("indexType", 0)
        axis = self.op_param.get("axis", 0)

        if index_type == 1:
            in_tensors[0] = in_tensors[0].index_add_(axis, in_tensors[1], in_tensors[2], alpha=in_tensors[3].item())
            return [in_tensors[0]]

        return []

    def test(self):
        ret = self.validate_param("indexType", "axis")
        if not ret:
            return
        self.execute()