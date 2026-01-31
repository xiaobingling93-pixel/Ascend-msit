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


class OpcheckSortOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        num = self.param.get("num", None)
        values, indices = torch.topk(in_tensors[0], k=num[0], largest=True)
        return [values, indices.int()]

    def test_3d_float(self):
        ret = self.validate_param("num")
        if not ret:
            return
        self.execute()