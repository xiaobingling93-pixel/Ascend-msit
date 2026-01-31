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
from enum import Enum
import torch
import torch_npu

from msit_llm.opcheck import operation_test
from msit_llm.common.log import logger


class ReduceType(Enum):
    REDUCE_UNDEFINED = 0
    REDUCE_MAX = 1
    REDUCE_MIN = 2
    REDUCE_SUM = 3


class OpcheckReduceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        op_type = self.op_param.get('reduceType', ReduceType.REDUCE_UNDEFINED.value)
        reduce_type_support_list = [
            ReduceType.REDUCE_MAX.value,
            ReduceType.REDUCE_MIN.value,
            ReduceType.REDUCE_SUM.value
        ]
        self.validate_int_range(op_type, reduce_type_support_list, "reduceType")
        axis = self.op_param.get('axis', None)
        if op_type == ReduceType.REDUCE_MAX.value:
            return [in_tensors[0].amax(axis)[0]]
        elif op_type == ReduceType.REDUCE_MIN.value:
            return [in_tensors[0].amin(axis)[0]]
        else:
            return [torch.sum(in_tensors[0], axis)]

    def test(self):
        ret = self.validate_param("reduceType", "axis")
        if not ret:
            return
        self.execute()