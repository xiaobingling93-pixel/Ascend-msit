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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict
from ..utils import get_handler, ErrorType, ErrorHandler


@dataclass
class CollectResult:
    data: Dict
    error_handler: ErrorHandler


class BaseCollector(ABC):
    def __init__(self, error_handler: ErrorHandler = None):
        self.error_handler = error_handler or get_handler(ErrorType.ERR_COLLECT)

    @abstractmethod
    def _collect_data(self) -> Dict:
        pass

    def collect(self) -> CollectResult:
        return CollectResult(self._collect_data(), self.error_handler)
