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


class InvalidParameterError(Exception):
    # Class-level constants for colors (only computed once)
    _COLORS = {
        'RED': "\033[1;31m",
        'YELLOW': "\033[1;33m",
        'CYAN': "\033[1;36m",
        'RESET': "\033[0m"
    }

    def __init__(self, parameter_name, caller_name, constraint, parameter_value):
        self.parameter_name = parameter_name
        self.caller_name = caller_name
        self.constraint = constraint
        self.parameter_value = parameter_value

        super().__init__(self._build_error_message())

    def _build_error_message(self):
        """Build the complete error message with colors and formatting"""
        c = self._COLORS

        indent = ' ' * 4
        message_parts = [
            f"{c['RED']}Parameter validation failed at{c['RESET']}",
            f"{c['CYAN']}Where:{c['RESET']}",
            f"{indent}Function: {c['YELLOW']}{self.caller_name}{c['RESET']}",
            f"{indent}Parameter: {c['YELLOW']}{self.parameter_name}{c['RESET']}",
            f"{indent}Received value: {c['RED']}{self.parameter_value!r}{c['RESET']}",
            "",
            f"{c['CYAN']}Requirement:{c['RESET']}",
            f"{indent}{self.constraint}",
            "",
            f"{c['CYAN']}Hint:{c['RESET']}",
            f"{indent}Check the last '{c['RED']}[F]{c['RESET']}' and "
            f"make sure {c['RED']}{self.parameter_value!r}{c['RESET']} is valid"
        ]

        return "\n".join(message_parts)

