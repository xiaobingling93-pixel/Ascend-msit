# -*- coding: utf-8 -*-
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
from enum import Enum, unique

from model_evaluation.common.enum.atc_err import AtcErr


@unique
class Rule(Enum):
    EVAL_ATC_SUCCESS = 0
    EVAL_ATC_UNSUPPORT_OP_ERR = 1
    EVAL_ATC_OTHER_ERR = 2

    @staticmethod
    def get_rule_with_atc_err(err: AtcErr):
        if err == AtcErr.SUCCESS:
            return Rule.EVAL_ATC_SUCCESS
        unsupported_op_err = [AtcErr.E19010, AtcErr.EZ0501, AtcErr.EZ3002, AtcErr.EZ3003, AtcErr.EZ9010]
        if err in unsupported_op_err:
            return Rule.EVAL_ATC_UNSUPPORT_OP_ERR

        return Rule.EVAL_ATC_OTHER_ERR
