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

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __repr__(self):
        return self.value


class ResultStatus:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

    def __init__(self, passed, severity):
        self._passed = passed
        self._severity = severity

    def __str__(self):
        if self._passed:
            status = "[OK]"
            color = self.GREEN
        else:
            if self._severity == Severity.HIGH:
                status = "[NOK]"
                color = self.RED
            elif self._severity == Severity.MEDIUM:
                status = "[WARNING]"
                color = self.YELLOW
            else:
                status = "[RECOMMEND]"
                color = self.CYAN
        return "{}{}{}".format(color, status, self.RESET)

    def __bool__(self):
        return self._passed

    @property
    def passed(self):
        return self._passed

    @property
    def severity(self):
        return self._severity


@dataclass(frozen=True)
class Result:
    key: str
    actual: str
    expected: str
    status: ResultStatus
    reason: str
