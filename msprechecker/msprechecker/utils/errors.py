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

import bisect
from enum import Enum
from functools import total_ordering
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class ErrorType(Enum):
    ERR_COLLECT = 0
    ERR_CHECK = 1
    ERR_COMPARE = 2
    ERR_UNKNOWN = 3


@total_ordering
class ErrorSeverity(Enum):
    ERR_HIGH = "high"
    ERR_MEDIUM = "medium"
    ERR_LOW = "low"

    @property
    def symbol(self):
        return {
            ErrorSeverity.ERR_HIGH: "NOK",
            ErrorSeverity.ERR_MEDIUM: "WARNING",
            ErrorSeverity.ERR_LOW: "RECOMMEND"
        }[self]

    @property
    def color_code(self):
        return {
            ErrorSeverity.ERR_HIGH: "\033[91m",
            ErrorSeverity.ERR_MEDIUM: "\033[93m",
            ErrorSeverity.ERR_LOW: "\033[96m"
        }[self]

    def __str__(self):
        reset = "\033[0m"
        return f"{self.color_code}[{self.symbol}]{reset}"
    
    _ORDER_MAP = {
        "low": 0,
        "medium": 1,
        "high": 2
    }

    def __gt__(self, other):
        order_map = self._ORDER_MAP.value
        if isinstance(other, ErrorSeverity):
            return order_map[self.value] > order_map[other.value]

        if isinstance(other, str):
            return order_map[self.value] > order_map[other.lower()]


class BaseError(ABC):
    def __init__(self, reason: str, severity: ErrorSeverity):
        if isinstance(severity, str):
            severity = ErrorSeverity(severity)

        self.reason = reason
        self.severity = severity


class CollectError(BaseError):
    def __init__(self, filename: str, function: str, lineno: int, what: str, reason: str, severity: ErrorSeverity):
        super().__init__(reason, severity)
        self.filename = filename
        self.function = function
        self.lineno = lineno
        self.what = what


class CheckError(BaseError):
    def __init__(self, path: str, actual: str, expected: str, reason: str, severity: ErrorSeverity):
        super().__init__(reason, severity)
        self.path = path
        self.actual = actual
        self.expected = expected
        


class ConfigError(CheckError):
    def __init__(self, path: str, actual: str, expected: str, reason: str, severity: ErrorSeverity, lineno: int = None, start_col: int = None, context_lines: Dict[int, str] = None):
        super().__init__(path, actual, expected, reason, severity)
        self.lineno = lineno
        self.start_col = start_col
        self.context_lines = context_lines


class CompareError(BaseError):
    def __init__(self, key, values, reason="", severity=ErrorSeverity.ERR_LOW):
        super().__init__(reason, severity)
        self.key = key
        self.values = values


class ErrorHandler(ABC):
    def __init__(self, *, severity: ErrorSeverity = None, type: str = ""):
        self._errors: List[BaseError] = []
        self._type = type
        self._severity = severity or ErrorSeverity.ERR_LOW
    
    def __iter__(self):
        return iter(self._errors)

    @abstractmethod
    def add_error(self, *args, **kwargs) -> None:
        """add error to handler"""
    
    @property
    def errors(self):
        return self._errors
    
    @property
    def severity(self):
        return self._severity
    
    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, another_type):
        self._type = another_type

    def empty(self) -> bool:
        return len(self._errors) == 0
    
    def extend(self, other_handler) -> None:
        if not isinstance(other_handler, ErrorHandler):
            raise TypeError
        
        self._errors.extend(other_handler.errors)
    
    def filter(self, severity: ErrorSeverity):
        return list(filter(lambda error: error.severity >= severity, self._errors))
    
    def clear(self) -> None:
        self._errors.clear()


class CollectErrorHandler(ErrorHandler):
    def add_error(self, filename: str, function: str, lineno: int, what: str, reason: str, severity: ErrorSeverity = ErrorSeverity.ERR_MEDIUM):
        if severity < self._severity:
            return

        self._errors.append(CollectError(
            filename=filename,
            function=function,
            lineno=lineno,
            what=what,
            reason=reason,
            severity=severity
        ))


class CheckErrorHandler(ErrorHandler):
    def add_error(self, path: str, actual: str, expected: str, reason: str, severity: ErrorSeverity):
        if severity < self._severity:
            return

        self._errors.append(CheckError(
            path=path,
            actual=actual,
            expected=expected,
            reason=reason,
            severity=severity
        ))


class ConfigErrorHandler(ErrorHandler):
    def __init__(self, severity: ErrorSeverity = None, file_lines: List[str] = None, key_mapping: Dict[str, Tuple[int, int]] = None, context_hierarchy: List[List[int]] = None, type: str = ""):
        super().__init__(severity=severity, type=type)
        self._file_lines = file_lines
        self._key_mapping = key_mapping
        self._sorted_key = sorted(key_mapping)
        self._context_hierarchy = context_hierarchy

    def add_error(self, path: str, actual: str, expected: str, reason: str, severity: ErrorSeverity):
        if severity < self._severity:
            return

        orig_lineno, shifted_lineno, start_col = self._find_lineno_and_col(path)
        path = path.split('.')[-1]
        context_lines = {context_lineno + 1: self._file_lines[context_lineno] for context_lineno in self._context_hierarchy[orig_lineno]}
        self._errors.append(ConfigError(
            path=path,
            actual=actual,
            expected=expected,
            reason=reason,
            severity=severity,
            lineno=shifted_lineno + 1,
            start_col=start_col,
            context_lines=context_lines
        ))
    
    def _find_lineno_and_col(self, path):
        if path in self._key_mapping:
            lineno, start_col = self._key_mapping[path]
            return lineno, lineno, start_col
        
        nearest_path = self._find_nearest_path(path)
        lineno, start_col = self._key_mapping[nearest_path]
        lineno_shift = (len(self._errors) + 1) / len(self._file_lines) # add floating point for ordering

        return lineno, lineno + lineno_shift, start_col
    
    def _find_nearest_path(self, path):
        path_pos = bisect.bisect_left(self._sorted_key, path)
        path = self._sorted_key[path_pos]
        if path_pos == 0:
            nearest_path = self._sorted_key[path]
        elif path_pos == len(self._sorted_key) - 1:
            nearest_path = self._sorted_key[path_pos - 1]
        else:
            candidate = self._sorted_key[path_pos]
            second_candidate = self._sorted_key[path_pos - 1]
            if abs(len(candidate) - len(path)) < abs(len(second_candidate) - len(path)):
                nearest_path = candidate
            else:
                nearest_path = second_candidate

        return nearest_path


class CompareErrorHandler(ErrorHandler):
    def add_error(self, key: str, values: Dict[str, str]):
        self._errors.append(CompareError(
            key=key,
            values=values
        ))


def get_handler(error_type: ErrorType) -> ErrorHandler:
    cmd_to_handler = {
        ErrorType.ERR_COLLECT: CollectErrorHandler,
        ErrorType.ERR_CHECK: CheckErrorHandler,
        ErrorType.ERR_COMPARE: CompareErrorHandler,
    }

    if not isinstance(error_type, ErrorType):
        raise TypeError

    if error_type not in cmd_to_handler:
        raise ValueError
    
    return cmd_to_handler[error_type]()
