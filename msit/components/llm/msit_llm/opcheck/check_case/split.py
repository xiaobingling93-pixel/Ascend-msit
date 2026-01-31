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

from msit_llm.opcheck import operation_test


class OpcheckAddOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        split_dim = self.op_param.get('splitDim', 0)
        split_num = self.op_param.get('splitNum', 2) # 等分次数，当前支持2或3
        split_size = self.op_param.get('splitSizes', []) 
        self.validate_int_range(split_num, [2, 3], "splitNum")
        if split_size:
            split_output = torch.split(in_tensors[0], split_size_or_sections=split_size, dim=split_dim)
        else:
            split_output = torch.chunk(in_tensors[0], chunks=split_num, dim=split_dim)
        return split_output

    def test(self):
        ret = self.validate_param("splitNum", "splitDim")
        if not ret:
            return
        self.execute()