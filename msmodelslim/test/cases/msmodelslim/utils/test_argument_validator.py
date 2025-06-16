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
from msmodelslim.utils.safe_utils import ArgumentValidator

class TestArgumentValidator:
    def test_init_with_allow_none(self):
        validator = ArgumentValidator(allow_none=True)
        assert validator.allow_none is True
        assert validator.validation_pipeline == []

    def test_init_without_allow_none(self):
        validator = ArgumentValidator(allow_none=False)
        assert validator.allow_none is False
        assert validator.validation_pipeline == []

    def test_validate_with_none_allowed(self):
        validator = ArgumentValidator(allow_none=True)
        validator.validate(None)  # Should not raise any exception

    def test_validate_with_value(self):
        validator = ArgumentValidator(allow_none=False)
        validator.add_validation_method(lambda x: None)  # Add a dummy validation method
        validator.validate("some_value")  # Should not raise any exception

    def test_add_validation_method_with_position(self):
        validator = ArgumentValidator(allow_none=False)
        method1 = lambda x: None
        method2 = lambda x: None
        validator.add_validation_method(method1)
        validator.add_validation_method(method2, position=0)
        assert validator.validation_pipeline == [method2, method1]

    def test_add_validation_method_with_target_method(self):
        validator = ArgumentValidator(allow_none=False)
        method1 = lambda x: None
        method2 = lambda x: None
        validator.add_validation_method(method1)
        validator.add_validation_method(method2, target_method=method1)
        assert validator.validation_pipeline == [method1, method2]

    def test_add_validation_method_without_position_or_target(self):
        validator = ArgumentValidator(allow_none=False)
        method1 = lambda x: None
        method2 = lambda x: None
        validator.add_validation_method(method1)
        validator.add_validation_method(method2)
        assert validator.validation_pipeline == [method1, method2]

    def test_delete_validation_method_with_position(self):
        validator = ArgumentValidator(allow_none=False)
        method1 = lambda x: None
        method2 = lambda x: None
        validator.add_validation_method(method1)
        validator.add_validation_method(method2)
        validator.delete_validation_method(position=0)
        assert validator.validation_pipeline == [method2]

    def test_delete_validation_method_with_method(self):
        validator = ArgumentValidator(allow_none=False)
        method1 = lambda x: None
        method2 = lambda x: None
        validator.add_validation_method(method1)
        validator.add_validation_method(method2)
        validator.delete_validation_method(method=method1)
        assert validator.validation_pipeline == [method2]

    def test_create_validation_pipeline(self):
        validator = ArgumentValidator(allow_none=False)
        method1 = lambda x: None
        method2 = lambda x: None
        validator._create_validation_pipeline(method1, method2)
        assert validator.validation_pipeline == [method1, method2]