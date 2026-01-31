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
PREFILL = "prefill"
DECODE = "decode"
DECODE_FILE_NAME = "decode_global_deployment"
PREFILL_FILE_NAME = "prefill_global_deployment" 

ALGORITHM_C2LB = "0"
ALGORITHM_SPECULATIVE_MOE_LEVEL_1 = "1"
ALGORITHM_DYNAMIC_C2LB = "2"
ALGORITHM_SPECULATIVE_MOE_LEVEL_2 = "3"
ALGORITHM_SPECULATIVE_MOE_LEVEL_1_MIXED = "4"
ALGORITHM_SPECULATIVE_MOE_LEVEL_2_MIXED = "5"

A2 = "a2"
A3 = "a3"

SUPPORTED_COMBINATIONS = {
    "a2": {"1", "3", "4", "5"},
    "a3": {"1", "3"},
}

SPECULATIVE_MOE_ALGORITHM = {
        ALGORITHM_SPECULATIVE_MOE_LEVEL_1,
        ALGORITHM_SPECULATIVE_MOE_LEVEL_2,
        ALGORITHM_SPECULATIVE_MOE_LEVEL_1_MIXED,
        ALGORITHM_SPECULATIVE_MOE_LEVEL_2_MIXED
    }