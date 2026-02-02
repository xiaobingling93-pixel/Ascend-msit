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

from .base import BaseValidator


class RangeValidator(BaseValidator):
    @staticmethod
    def validate(actual_value, expected_value):
        if actual_value is None or not isinstance(actual_value, (int, float)):
            return False

        if isinstance(expected_value, str):
            lower_bound, upper_bound, left_inclusive, right_inclusive = \
                RangeValidator._parse_expected_value_str(expected_value)
        elif isinstance(expected_value, list):
            lower_bound, upper_bound, left_inclusive, right_inclusive = \
                RangeValidator._parse_expected_value_list(expected_value)
        else:
            raise ValueError(f"Expected a string or list, got {type(expected_value).__name__}")

        try:
            lower_bound = float(lower_bound)
            upper_bound = float(upper_bound)
        except ValueError as e:
            raise ValueError(f"Expected numeric values, got {expected_value}") from e

        if lower_bound > upper_bound:
            raise ValueError(f"Lower bound must be less than or equal to upper bound, got {expected_value}")

        left = lower_bound <= actual_value if left_inclusive else lower_bound < actual_value
        right = actual_value <= upper_bound if right_inclusive else actual_value < upper_bound

        return left and right

    @staticmethod
    def _parse_expected_value_str(expected_value):
        split_field = expected_value.split(",")
        if len(split_field) != 2:
            raise ValueError(f"Expected a string with two comma-separated values, got {expected_value}")
        lower_bound, upper_bound = split_field
        if not lower_bound.startswith(("(", "[")) or not upper_bound.endswith((")", "]")):
            raise ValueError(f"Expected a string starting with '(' or '[' and ending with ')' or ']', "
                             f"got {expected_value}")

        left_inclusive = lower_bound.startswith("[")
        right_inclusive = upper_bound.endswith("]")

        lower_bound = lower_bound.strip("[( ")
        upper_bound = upper_bound.strip(" )]")
        return lower_bound, upper_bound, left_inclusive, right_inclusive

    @staticmethod
    def _parse_expected_value_list(expected_value):
        if len(expected_value) != 2:
            raise ValueError(f"Expected a list with two elements, got {expected_value}")
        lower_bound, upper_bound = expected_value
        left_inclusive = True
        right_inclusive = True
        return lower_bound, upper_bound, left_inclusive, right_inclusive
