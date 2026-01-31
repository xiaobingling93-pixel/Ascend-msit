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


@unique
class AtcErr(Enum):
    SUCCESS = 0  # atc execute success
    UNKNOWN = -1  # unknown error, include other atc errcode
    EZ0501 = 1  # IR for the operator type is not registered
    EZ3002 = 2  # The operator type is unsupported in the operator information library due to specification mismatch.
    EZ3003 = 3  # The operator is not supported
    EZ9010 = 4  # No parser is registered for Op
    E19010 = 5  # No parser is registered for Op
