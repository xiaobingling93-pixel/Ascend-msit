# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import argparse
from enum import Enum
from abc import ABC, abstractmethod


class CommandType(Enum):
    CMD_PRECHECK = "precheck"
    CMD_DUMP = "dump"
    CMD_COMPARE = "compare"
    CMD_RUN = "run"
    CMD_INSPECT = "inspect"


class CommandStrategy(ABC):
    @staticmethod
    @abstractmethod
    def execute(args: argparse.Namespace) -> int:
        """Execute the command strategy"""
        pass
