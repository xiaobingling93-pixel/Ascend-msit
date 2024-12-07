# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

from unittest.mock import MagicMock, patch

import pytest

from components.utils.check.func_wrapper import FuncWrapper, validate_params


@pytest.fixture(scope="module", autouse=True)
def setup():
    # Setup any mock objects or state needed for tests
    pass


class TestFuncWrapper:

    @staticmethod
    def test_is_function_param_valid_given_no_rules_when_called_then_true():
        func_wrapper = FuncWrapper({})
        result = func_wrapper.is_function_param_valid(False)
        assert result is True

    @staticmethod
    def test_is_function_param_valid_given_invalid_kwarg_when_called_then_false_with_message():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = False
        func_wrapper = FuncWrapper({'test_param': check_rule_mock})
        result = func_wrapper.is_function_param_valid(False, test_param='invalid')
        assert isinstance(result, tuple)
        assert result[0] is False
        assert 'test_param is invalid.' in result[1]

    @staticmethod
    def test_is_function_param_valid_given_valid_kwarg_when_called_then_true():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = True
        func_wrapper = FuncWrapper({'test_param': check_rule_mock})
        result = func_wrapper.is_function_param_valid(False, test_param='valid')
        assert result is True

    @staticmethod
    def test_is_function_param_valid_given_invalid_var_keyword_when_called_then_false_with_message():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = False
        func_wrapper = FuncWrapper({})
        func_wrapper.var_keyword_rule = check_rule_mock
        result = func_wrapper.is_function_param_valid(False, unexpected_param='invalid')
        assert isinstance(result, tuple)
        assert result[0] is False
        assert 'unexpected_param is invalid.' in result[1]

    @staticmethod
    def test_is_function_param_valid_given_valid_var_keyword_when_called_then_true():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = True
        func_wrapper = FuncWrapper({})
        func_wrapper.var_keyword_rule = check_rule_mock
        result = func_wrapper.is_function_param_valid(False, unexpected_param='valid')
        assert result is True

    @staticmethod
    def test_is_function_param_valid_given_invalid_args_when_called_then_false_with_message():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = False
        func_wrapper = FuncWrapper({'param1': check_rule_mock})
        func_wrapper.args_names = ['param1']
        result = func_wrapper.is_function_param_valid(False, 'invalid')
        assert isinstance(result, tuple)
        assert result[0] is False
        assert 'param1 is invalid.' in result[1]

    @staticmethod
    def test_is_function_param_valid_given_valid_args_when_called_then_true():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = True
        func_wrapper = FuncWrapper({'param1': check_rule_mock})
        func_wrapper.args_names = ['param1']
        result = func_wrapper.is_function_param_valid(False, 'valid')
        assert result is True

    @staticmethod
    def test_parse_function_given_function_with_params_when_called_then_parses_correctly():
        def test_func(param1, param2=None, **kwargs): pass
        func_wrapper = FuncWrapper({})
        func_wrapper.parse_function(test_func)
        assert func_wrapper.args_names == ['param1', 'param2']
        assert 'param1' in func_wrapper.args_name_set
        assert 'param2' in func_wrapper.args_name_set

    @staticmethod
    def test_parse_function_given_function_with_var_keyword_when_called_then_sets_var_keyword_rule():
        def test_func(**kwargs): pass
        func_wrapper = FuncWrapper({'kwargs': MagicMock()})
        func_wrapper.parse_function(test_func)
        assert func_wrapper.var_keyword_rule is not None

    @staticmethod
    def test_create_wrapper_given_ret_value_and_to_raise_when_called_then_returns_decorator():
        func_wrapper = FuncWrapper({})
        decorator = func_wrapper.create_wrapper('ret_value', True)
        assert callable(decorator)

    @staticmethod
    def test_create_wrapper_given_function_when_wrapped_then_validates_params():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = True
        func_wrapper = FuncWrapper({'param1': check_rule_mock})

        @func_wrapper.create_wrapper('default_return', False)
        def test_func(param1): return 'original_return'

        result = test_func('valid')
        assert result == 'original_return'

    @staticmethod
    def test_create_wrapper_given_invalid_params_when_wrapped_then_returns_ret_value():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = False
        logger_mock = MagicMock()
        func_wrapper = FuncWrapper({'param1': check_rule_mock})

        @func_wrapper.create_wrapper('default_return', False, logger_mock)
        def test_func(param1): return 'original_return'

        result = test_func('invalid')
        logger_mock.error.assert_called_once()
        assert result == 'default_return'

    @staticmethod
    def test_to_return_given_ret_value_when_called_then_creates_wrapper():
        func_wrapper = FuncWrapper({})
        wrapper = func_wrapper.to_return('ret_value')
        assert callable(wrapper)

    @staticmethod
    def test_to_raise_given_called_when_called_then_creates_wrapper():
        func_wrapper = FuncWrapper({})
        wrapper = func_wrapper.to_raise()
        assert callable(wrapper)

    @staticmethod
    def test_validate_params_given_check_rules_when_called_then_returns_func_wrapper():
        check_rules = {'param1': MagicMock()}
        result = validate_params(**check_rules)
        assert isinstance(result, FuncWrapper)
        assert result.check_rules == check_rules
import pytest
from unittest.mock import MagicMock, patch
import logging

from components.utils.check.func_wrapper import validate_params, FuncWrapper
from components.utils.check.number_checker import NumberChecker
from components.utils.check.dict_checker import DictChecker
from components.utils.check.string_checker import StringChecker


@pytest.mark.parametrize("a_value, b_value, value", [
    ("55", {"a": 8}, ValueError),
    (3, ["a", 8], ValueError),
])
def test_func_wrapper_to_raise_when_fail(a_value, b_value, value):
    @validate_params(a=NumberChecker().is_int(), b=DictChecker().is_dict()).to_raise()
    def basic_func(a: int, b: dict):
        return f"{a} is an int and {b} is a dict"

    with pytest.raises(value):
        basic_func(a_value, b_value)


@pytest.mark.parametrize("a_value, b_value, value", [
    (3, {"a": 8}, "3 is an int and {\'a\': 8} is a dict"),
])
def test_func_wrapper_to_raise_when_pass(a_value, b_value, value):
    @validate_params(a=NumberChecker().is_int(), b=DictChecker().is_dict()).to_raise()
    def basic_func(a: int, b: dict):
        return f"{a} is an int and {b} is a dict"

    res = basic_func(a_value, b_value)
    assert res == value


@pytest.mark.parametrize("a_value, b_value, value", [
    (3, {"a": 8}, "3 is an int and {\'a\': 8} is a dict"),
    ("55", {"a": 8}, False),
    (3, ["a", 8], False),
])
def test_func_wrapper_to_return(a_value, b_value, value):
    @validate_params(a=NumberChecker().is_int(), b=DictChecker().is_dict()).to_return(False)
    def basic_func(a: int, b: dict):
        return f"{a} is an int and {b} is a dict"

    res = basic_func(a_value, b_value)
    assert res == value


@pytest.mark.parametrize("a_value, b_value, expected_exception", [
    (None, {"a": 8}, ValueError),  # Test with None
    ("100", {"a": 8}, ValueError),  # Test with string that could be an int
    # (0, {}, ValueError),           # Test with boundary numeric value and empty dict
    (999999999, {"a": 8}, None),  # Test with large int
])
def test_func_wrapper_to_raise_various_inputs(a_value, b_value, expected_exception):
    @validate_params(a=NumberChecker().is_int(), b=DictChecker().is_dict()).to_raise()
    def basic_func(a: int, b: dict):
        return f"{a} is an int and {b} is a dict"

    if expected_exception:
        with pytest.raises(expected_exception):
            basic_func(a_value, b_value)
    else:
        assert basic_func(a_value, b_value) == f"{a_value} is an int and {b_value} is a dict"


@pytest.mark.parametrize("a_value, b_value, expected_output", [
    # (0, {}, False),  # Test with zero and empty dict
    (-1, None, False),  # Test with negative number and None
])
def test_func_wrapper_to_return_empty_inputs(a_value, b_value, expected_output):
    @validate_params(a=NumberChecker().is_int(), b=DictChecker().is_dict()).to_return(False)
    def basic_func(a: int, b: dict):
        return f"{a} is an int and {b} is a dict"

    res = basic_func(a_value, b_value)
    assert res == expected_output


def test_func_wrapper_no_rules():
    @validate_params().to_return("No checks")
    def basic_func():
        return "No checks"

    assert basic_func() == "No checks"


@pytest.mark.parametrize("b_value, expected_output", [
    ({"a": {"nested": 123}}, "3 is an int and {'a': {'nested': 123}} is a dict"),
    # ({"a": {"nested": None}}, False),
])
def test_func_wrapper_nested_dicts(b_value, expected_output):
    @validate_params(a=NumberChecker().is_int(), b=DictChecker().is_dict()).to_return(False)
    def basic_func(a: int, b: dict):
        return f"{a} is an int and {b} is a dict"

    res = basic_func(3, b_value)
    assert res == expected_output


def test_func_wrapper_valid_params():
    mock_number_checker = MagicMock(spec=NumberChecker)
    mock_number_checker.check.return_value = True

    mock_string_checker = MagicMock(spec=StringChecker)
    mock_string_checker.check.return_value = True

    check_rules = {
        'age': mock_number_checker,
        'name': mock_string_checker
    }

    @validate_params(**check_rules).to_return(ret_value="Success")
    def sample_function(age, name):
        return "Success"

    result = sample_function(age=30, name="Alice")

    assert result == "Success"
    mock_number_checker.check.assert_called_once_with(30, will_raise=False)
    mock_string_checker.check.assert_called_once_with("Alice", will_raise=False)


def test_func_wrapper_invalid_params_return():
    # Arrange
    mock_number_checker = MagicMock(spec=NumberChecker)
    mock_number_checker.check.return_value = False

    check_rules = {
        'age': mock_number_checker
    }

    @validate_params(**check_rules).to_return(ret_value="Invalid Age")
    def sample_function(age):
        return "Success"

    result = sample_function(age=-1)

    assert result == "Invalid Age"
    mock_number_checker.check.assert_called_once_with(-1, will_raise=False)


def test_func_wrapper_invalid_params_raise():
    mock_number_checker = MagicMock(spec=NumberChecker)
    mock_number_checker.check.return_value = False

    check_rules = {
        'age': mock_number_checker
    }

    @validate_params(**check_rules).to_raise()
    def sample_function(age):
        return "Success"

    sample_function(age=-1)
    mock_number_checker.check.assert_called_once_with(-1, will_raise=True)


def test_func_wrapper_with_logger():
    # Arrange
    mock_number_checker = MagicMock(spec=NumberChecker)
    mock_number_checker.check.return_value = False

    check_rules = {
        'age': mock_number_checker
    }

    mock_logger = MagicMock(spec=logging.Logger)

    @validate_params(**check_rules).to_return(ret_value="Invalid Age", logger=mock_logger)
    def sample_function(age):
        return "Success"

    result = sample_function(age=-1)

    assert result == "Invalid Age"
    mock_number_checker.check.assert_called_once_with(-1, will_raise=False)
    mock_logger.error.assert_called_once_with("(False, 'age is invalid. False')")


def test_func_wrapper_var_keyword_args():
    mock_number_checker = MagicMock(spec=NumberChecker)
    mock_number_checker.check.return_value = True

    mock_dict_checker = MagicMock(spec=DictChecker)
    mock_dict_checker.check.return_value = False

    check_rules = {
        'age': mock_number_checker,
        'kwargs': mock_dict_checker
    }

    @validate_params(**check_rules).to_return(ret_value="Invalid Kwargs")
    def sample_function(age, **kwargs):
        return "Success"

    result = sample_function(age=25, extra="data")

    assert result == "Invalid Kwargs"
    mock_number_checker.check.assert_called_once_with(25, will_raise=False)
    mock_dict_checker.check.assert_called_once_with("data", will_raise=False)


def test_create_wrapper_no_kwargs():
    mock_number_checker = MagicMock(spec=NumberChecker)
    mock_number_checker.check.return_value = True

    check_rules = {
        'age': mock_number_checker
    }

    @validate_params(**check_rules).to_return(ret_value="Valid")
    def sample_function(age):
        return "Valid"

    result = sample_function(40)

    assert result == "Valid"
    mock_number_checker.check.assert_called_once_with(40, will_raise=False)


def test_multiple_parameters():
    mock_number_checker = MagicMock(spec=NumberChecker)
    mock_number_checker.check.side_effect = [True, False]

    check_rules = {
        'age': mock_number_checker,
        'score': mock_number_checker
    }

    @validate_params(**check_rules).to_return(ret_value="Invalid Score")
    def sample_function(age, score):
        return "Success"

    result = sample_function(age=25, score=-10)

    assert result == "Invalid Score"
    assert mock_number_checker.check.call_count == 2
    mock_number_checker.check.assert_any_call(25, will_raise=False)
    mock_number_checker.check.assert_any_call(-10, will_raise=False)
