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
import numpy as np

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class OpcheckMatmulOperation(operation_test.OperationTest):
    def golden_flp(self, transpose_a: bool, transpose_b: bool, in_tensor_0, in_tensor_1):
        if transpose_a:
            if len(in_tensor_0.shape) == 2:
                in_tensor_0 = torch.permute(in_tensor_0, (1, 0))
            if len(in_tensor_0.shape) == 3:
                in_tensor_0 = torch.permute(in_tensor_0, (0, 2, 1))
        if len(in_tensor_1.shape) == 4:
            in_tensor_1 = torch.permute(in_tensor_1, (0, 2, 1, 3))
            if in_tensor_1.shape[0] == 1:
                in_tensor_1 = in_tensor_1.reshape(in_tensor_1.shape[1], in_tensor_1.shape[2] * in_tensor_1.shape[3])
            else:
                in_tensor_1 = in_tensor_1.reshape(in_tensor_1.shape[0], in_tensor_1.shape[1],
                                                  in_tensor_1.shape[2] * in_tensor_1.shape[3])
        if transpose_b:
            if len(in_tensor_1.shape) == 2:
                in_tensor_1 = torch.permute(in_tensor_1, (1, 0))
            if len(in_tensor_1.shape) == 3:
                in_tensor_1 = torch.permute(in_tensor_1, (0, 2, 1))
        golden_result = torch.matmul(in_tensor_0, in_tensor_1)
        return golden_result

    def golden_calc(self, in_tensors):
        transpose_a = self.op_param.get("transposeA", None)
        transpose_b = self.op_param.get("transposeB", None)
        golden_result = self.golden_flp(transpose_a, transpose_b, in_tensors[0], in_tensors[1])
        golden_result = torch.tensor(golden_result).half()
        return [golden_result]

    def test(self):
        ret = self.validate_param("transposeA", "transposeB")
        if not ret:
            return
        self.execute()