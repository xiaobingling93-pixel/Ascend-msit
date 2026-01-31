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


class OpcheckLinearSparseOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        soc_version = self.get_soc_version()
        if soc_version == 'Ascend310P':
            in_tensors[1] = self.convert_data_format(in_tensors[1])
            logger_text = "The result of this case is unreliable on Ascend310P!"
            logger.info(logger_text)

        transpose_a = self.op_param.get("transposeA", False)
        transpose_b = self.op_param.get("transposeB", True)

        x = in_tensors[0]
        weight = in_tensors[1]
        bias = in_tensors[2] if len(in_tensors) >= 3 else None # 当has_bias = true时才输入
        deq_scale = in_tensors[3] if len(in_tensors) >= 4 else None # 反量化的scale，量化场景下才输入

        if transpose_a:
            x = torch.transpose(x, 0, 1) if len(x.shape) == 2 else torch.transpose(x, 1, 2)

        if len(weight.shape) == 4:
            weight = torch.transpose(weight, 1, 2)
            if weight.shape[0] == 1:
                weight = weight.reshape(weight.shape[1], weight.shape[2] * weight.shape[3])
            else:
                weight = weight.reshape(weight.shape[0], weight.shape[1], weight.shape[2] * weight.shape[3])

        if transpose_b:
            weight = torch.transpose(weight, 0, 1) if len(weight.shape) == 2 else torch.transpose(weight, 1, 2)

        golden_result = torch.matmul(x, weight)
        if bias is not None:
            golden_result += bias
        if deq_scale is not None:
            golden_result *= deq_scale

        return [golden_result.type(torch.float16)]

    def test(self):
        ret = self.validate_param("transposeA", "transposeB")
        if not ret:
            return

        tilingk = self.op_param.get('tilingK', 1)
        tilingn = self.op_param.get('tilingN', 1)
        logger_text = f"tilingK: {tilingk}, tilingN: {tilingn} \nOnly 8 is supported!"
        logger.debug(logger_text)

        self.execute()