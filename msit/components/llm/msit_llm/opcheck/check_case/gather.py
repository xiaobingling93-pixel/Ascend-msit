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


class OpcheckGatherOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        in_tensors = [tensor.cpu() for tensor in in_tensors]
        axis = self.op_param.get("axis", 0)
        batch_dims = self.op_param.get("batchDims", 0)
        if batch_dims == 0:
            if axis == 0:
                if in_tensors[0].ndim == 2 and in_tensors[1].ndim == 2:
                    embedding = torch.nn.Embedding(in_tensors[0].shape[0], in_tensors[0].shape[1])
                    embedding.weight.data.copy_(in_tensors[0])
                    embedding.weight.requires_grad = False
                    golden_result = embedding(in_tensors[1]).detach()
                    return [golden_result]
            output_size = []
            dim0 = 1
            for i in range(0, axis):
                output_size.append(in_tensors[0].shape[i])
                dim0 *= in_tensors[0].shape[i]
            dim1 = in_tensors[0].shape[axis]
            for i in range(0, in_tensors[1].ndim):
                output_size.append(in_tensors[1].shape[i])
            dim2 = 1
            for i in range(axis + 1, in_tensors[0].ndim):
                output_size.append(in_tensors[0].shape[i])
                dim2 *= in_tensors[0].shape[i]
            logger_text = f"output_size: {output_size}"
            logger.debug(logger_text)
            input_flatten = in_tensors[0].clone().reshape(-1)
            indices_flatten = in_tensors[1].clone().reshape(-1)
            golden_result = input_flatten.reshape([dim0, dim1, dim2])[:, indices_flatten]
        elif batch_dims > 0:
            golden_result = torch.gather(in_tensors[0], axis, in_tensors[1])
        else:
            logger_text = f"The value of batchDims is invalid: {batch_dims}. It should be 0 or a positive integer."
            logger.error(logger_text)
        return [golden_result]

    def test(self):
        ret = self.validate_param("axis", "batchDims")
        if not ret:
            return
        self.execute()