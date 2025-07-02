# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC, abstractmethod
from enum import Enum, auto


class ConstraintStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    SKIPPED = auto()
    
    @property
    def symbol(self):
        return {
            ConstraintStatus.SUCCESS: "T",
            ConstraintStatus.FAILURE: "F",
            ConstraintStatus.SKIPPED: "S"
        }[self]
    
    @property
    def color_code(self):
        return {
            ConstraintStatus.SUCCESS: "\033[1;32m",
            ConstraintStatus.FAILURE: "\033[1;31m",
            ConstraintStatus.SKIPPED: "\033[1;33m"
        }[self]
    
    def colored_symbol(self):
        return f"{self.color_code}{self.symbol}\033[0m"


class BaseConstraint(ABC):
    def __init__(self, *, description=None):
        self.description = "" if description is None else description 
        self.status = ConstraintStatus.SKIPPED

    def __str__(self):
        status_display = self.status.colored_symbol()
        return f"{self.description}: [{status_display}]"
    
    def __and__(self, other):
        from .logic import AndConstraint
        from .constraint_builder import make_constraint 
        return AndConstraint(self, make_constraint(other))

    def __or__(self, other):
        from .logic import OrConstraint
        from .constraint_builder import make_constraint   
        return OrConstraint(self, make_constraint(other))
    
    def __invert__(self):
        from .logic import NotConstraint
        return NotConstraint(self)

    def is_satisfied_by(self, val):
        ret = self._is_satisfied_by(val)
        self.status = ConstraintStatus.SUCCESS if ret else ConstraintStatus.FAILURE
        return ret

    @abstractmethod
    def _is_satisfied_by(self, path):
        pass
    

class BasePathConstraint(BaseConstraint):
    def __init__(self, *, description=None):
        super().__init__(description=description)

    @abstractmethod
    def _is_satisfied_by(self, path):
        pass

    def _get_path_stat(self, path: str):
        st = None
        # does not support file descriptor
        if not isinstance(path, (str, os.PathLike)):
            return st
        
        try:
            st = os.stat(path, follow_symlinks=False)
        except (OSError, ValueError):
            pass
        return st
