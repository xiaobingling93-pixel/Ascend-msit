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
from components.utils.util import safe_get


class OpcheckSetValueOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        starts = self.op_param.get("starts", None)
        ends = self.op_param.get("ends", None)
        strides = self.op_param.get("strides", None)
        golden_result = [safe_get(in_tensors, 0).clone(), safe_get(in_tensors, 1).clone()]
        for i, _ in enumerate(starts):
            self.validate_int_range(strides[i], [1], "strides") # 当前仅支持strides为全1
            start = safe_get(starts, i)
            end = safe_get(ends, i)
            stride = safe_get(strides, i)
            golden_result[0][start:end:stride].copy_(safe_get(in_tensors, 1))
        return golden_result

    def test(self):
        ret = self.validate_param("starts", "ends", "strides")
        if not ret:
            return
        self.execute()