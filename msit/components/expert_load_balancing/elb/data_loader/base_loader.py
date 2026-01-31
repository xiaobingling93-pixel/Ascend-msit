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


class DataType(Enum):
    MINDIE_SUMED_CSV = 0
    MINDIE_SPLITED_CSV = 1
    MINDIE_SPLITED_CSV_WITH_TOPK = 2
    VLLM_SUMED_TENSOR = 3
    UNKNOWN_TYPE = 999


class BaseDataLoader:
    def __init__(self, input_args):
        self.input_args = input_args
        self.data_type = DataType.UNKNOWN_TYPE
