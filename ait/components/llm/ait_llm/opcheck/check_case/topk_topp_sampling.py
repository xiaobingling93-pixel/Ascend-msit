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

from ctypes import CDLL
import torch
import torch_npu
import torch.nn as nn

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckToppOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        rand_seed = self.op_param.get('rand_seed', None)
        topk = self.op_param.get('topk', None)
        libc = CDLL("libc.so.6")
        libc.srand(rand_seed)
        rand_list = [libc.rand() / 0x7fffffff for i in range(64)]

        probs = in_tensors[0]
        topp = in_tensors[1]
        probs_sorted = torch.sort(probs, axis=-1)[..., ::-1][..., :topk]
        try:
            probs_div_sorted = probs_sorted / topp
        except ZeroDivisionError as e:
            raise e   
        indices_sorted = torch.argsort(-probs, kind='mergesort', axis=-1)[..., :topk]
        probs_sorted_sumed = torch.cumsum(probs_sorted, axis=-1)
        mask = torch.zeros_like(probs_sorted_sumed, dtype=torch.int32)
        mask[probs_sorted_sumed <= topp] = 1
        probs_div_sorted *= mask
        probs_div_sorted_sumed = torch.cumsum(probs_div_sorted, axis=-1)
        multinomial_probs = probs_div_sorted_sumed < rand_list[0]
        first_true_indeces = torch.argmax(~multinomial_probs, axis=-1)
        for i in range(probs.shape[0]):
            multinomial_probs[i, first_true_indeces[i]:] = False
        indices_sorted_sampled = torch.sum(multinomial_probs, axis=-1, keepdims=True)
        indices_sorted_sampled[indices_sorted_sampled >= topk] = 0
        indices_sampled = torch.take_along_axis(indices_sorted, indices_sorted_sampled, axis=-1)
        probs_sampled = torch.take_along_axis(probs_sorted, indices_sorted_sampled, axis=-1)
        return [indices_sampled, probs_sampled]

    def test(self):
        ret = self.validate_param("rand_seed", "topk")
        if not ret:
            return
        self.execute()