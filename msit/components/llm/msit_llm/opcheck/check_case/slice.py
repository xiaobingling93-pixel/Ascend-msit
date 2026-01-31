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


class OpcheckSliceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        offset_list = self.op_param.get('offsets', None)
        size_list = self.op_param.get('size', None)
        for index, offset in enumerate(offset_list):
            offset_list[index] = offset if offset >= 0 else offset + in_tensors[0].shape[index]
        for index, size in enumerate(size_list):
            size_list[index] = size if size != -1 else in_tensors[0].shape[index] - offset_list[index]
        self.validate_int_range(len(offset_list), [2, 3, 4], "len(offsets)")
        if len(offset_list) == 2:
            return [in_tensors[0][offset_list[0]: offset_list[0] + size_list[0], 
                    offset_list[1]: offset_list[1] + size_list[1]]]
        elif len(offset_list) == 3:
            return [in_tensors[0][offset_list[0]: offset_list[0] + size_list[0], 
                    offset_list[1]: offset_list[1] + size_list[1], offset_list[2]: offset_list[2] + size_list[2]]]
        else:
            return [in_tensors[0][offset_list[0]: offset_list[0] + size_list[0], 
                    offset_list[1]: offset_list[1] + size_list[1], offset_list[2]: offset_list[2] + size_list[2], 
                    offset_list[3]: offset_list[3] + size_list[3]]]

    def test(self):
        ret = self.validate_param("offsets", "size")
        if not ret:
            return
        self.execute()