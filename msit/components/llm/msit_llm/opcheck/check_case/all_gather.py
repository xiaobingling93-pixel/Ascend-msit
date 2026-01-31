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
from msit_llm.common.log import logger


class OpcheckAllGatherOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        new_in_tensors = self.get_new_in_tensors()
        golden_result = torch.stack(new_in_tensors, dim=0)
        return [golden_result]        

    def test_all_gather(self):
        if self.pid is None:
            logger_text = "Cannot get a valid pid, AllGatherOperation is not supported!"
            logger.error(logger_text)
            return

        ret = self.validate_param("rank", "rankRoot", "rankSize")
        if not ret:
            return

        self.execute()