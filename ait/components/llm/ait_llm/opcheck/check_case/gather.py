# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckGatherOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        axis = self.op_param.get("axis", None)
        if axis == 0:
            if in_tensors[0].ndim == 2 and in_tensors[0].ndim == 2:
                embedding = torch.nn.Embedding(in_tensors[0].shape[0], in_tensors[0].shape[1])
                embedding.weight.data.copy_(in_tensors[0])
                embedding.weight.requires_grad = False
                golden_result = embedding(in_tensors[1].cpu()).detach()
                return [golden_result.npu()]
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
        input_flatten = in_tensors[0].clone().reshape(-1)
        indices_flatten = in_tensors[1].clone().reshape(-1)
        logger_text = f"output_size: {output_size}"
        logger.debug(logger_text)
        golden_result_np = torch.zeros(output_size, dtype=torch.float16).reshape(-1).numpy()
        idx = 0
        for i in range(0, dim0):
            input_idx = i * dim1 * dim2
            for indice in indices_flatten:
                for k in range(0, dim2):
                    golden_result_np[idx] = input_flatten[input_idx + indice * dim2 + k]
                    idx += 1
        golden_result = torch.from_numpy(golden_result_np).reshape(output_size)
        return [golden_result.npu()]

    def test(self):
        ret = self.validate_param("axis")
        if not ret:
            return
        self.execute()