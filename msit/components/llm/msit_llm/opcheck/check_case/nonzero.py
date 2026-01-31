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


class OpcheckNonzeroOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        num_non_negative = torch.count_nonzero(in_tensors[0])
        padding_num = in_tensors[0].numel() - num_non_negative
        padding = torch.zeros(len(in_tensors[0].shape), padding_num)
        result = torch.stack(list(torch.nonzero(in_tensors[0], as_tuple=True)))
        result = torch.concat((result, padding), dim=1).long()

        return [result, torch.tensor(num_non_negative).long()]

    def test(self):
        self.execute()