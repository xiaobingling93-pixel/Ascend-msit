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
import ctypes
import torch
import torch_npu

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger
from components.utils.util import safe_get


class OpcheckMultinomialOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        samples = self.op_param.get("numSamples", 1)
        rand_seed = self.op_param.get("randSeed", 0)
        input0 = in_tensors[0]
        libc = ctypes.CDLL("libc.so.6")
        libc.srand(rand_seed)
        rand_list = [libc.rand() / 0x7fffffff for i in range(64)]
        ret = torch.zeros(size=(input0.shape[0], samples))

        sum_list = torch.cumsum(input0, axis=-1)
        iter_list = [(j, i) 
                    for j in range(input0.shape[0]) 
                    for i in range(input0.shape[1])]
        for z in range(samples):
            for j, i in iter_list:
                if (sum_list[j][i] > safe_get(rand_list, z)):
                    ret[j][z] = i
                    break
        return [ret.contiguous()]

    def test(self):
        ret = self.validate_param("numSamples", "randSeed")
        if not ret:
            return
        self.execute()