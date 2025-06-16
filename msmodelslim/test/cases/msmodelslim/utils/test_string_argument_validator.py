# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import pytest
import argparse
from msmodelslim.utils.safe_utils import StringArgumentValidator

class TestStringArgumentValidator:
    def test_init_with_default_values(self):
        validator = StringArgumentValidator()
        assert validator.min_length == 0
        assert validator.max_length == float('inf')
        assert validator.allow_none is False

    def test_init_with_custom_values(self):
        validator = StringArgumentValidator(min_length=5, max_length=10, allow_none=True)
        assert validator.min_length == 5
        assert validator.max_length == 10
        assert validator.allow_none is True

    def test_validate_type_with_string(self):
        StringArgumentValidator.validate_type("valid_string")  # Should not raise any exception

    def test_validate_type_with_non_string(self):
        with pytest.raises(argparse.ArgumentTypeError):
            StringArgumentValidator.validate_type(123)

    def test_validate_length_with_valid_length(self):
        validator = StringArgumentValidator(min_length=5, max_length=10)
        validator.validate_length("valid")  # Should not raise any exception

    def test_validate_length_with_min_length(self):
        validator = StringArgumentValidator(min_length=5, max_length=10)
        validator.validate_length("12345")  # Should not raise any exception

    def test_validate_length_with_max_length(self):
        validator = StringArgumentValidator(min_length=5, max_length=10)
        validator.validate_length("1234567890")  # Should not raise any exception

    def test_validate_length_with_length_less_than_min(self):
        validator = StringArgumentValidator(min_length=5, max_length=10)
        with pytest.raises(argparse.ArgumentTypeError):
            validator.validate_length("1234")

    def test_validate_length_with_length_greater_than_max(self):
        validator = StringArgumentValidator(min_length=5, max_length=10)
        with pytest.raises(argparse.ArgumentTypeError):
            validator.validate_length("12345678901")

    def test_create_validation_pipeline(self):
        validator = StringArgumentValidator(min_length=5, max_length=10)
        validator.create_validation_pipeline()
        assert len(validator.validation_pipeline) == 2
        assert callable(validator.validation_pipeline[0])
        assert callable(validator.validation_pipeline[1])