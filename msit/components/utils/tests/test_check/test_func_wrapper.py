import pytest

from components.utils.check.func_wrapper import validate_params
from components.utils.check.number_checker import NumberChecker
from components.utils.check.dict_checker import DictChecker


@pytest.mark.parametrize("a_value, b_value, value", [
    ("55", {"a": 8}, ValueError),
    (3, ["a", 8], ValueError),
])
def test_func_wrapper_to_raise_when_fail(a_value, b_value, value):
    check_rules = {"a": NumberChecker().is_int(), "b": DictChecker().is_dict()}

    @validate_params(check_rules).to_raise()
    def basic_func(a: int, b: dict):
        return f"{a} is an int and {b} is a dict"

    with pytest.raises(value):
        basic_func(a_value, b_value)


@pytest.mark.parametrize("a_value, b_value, value", [
    (3, {"a": 8}, "3 is an int and {\'a\': 8} is a dict"),
])
def test_func_wrapper_to_raise_when_pass(a_value, b_value, value):
    check_rules = {"a": NumberChecker().is_int(), "b": DictChecker().is_dict()}

    @validate_params(check_rules).to_raise()
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
    check_rules = {"a": NumberChecker().is_int(), "b": DictChecker().is_dict()}

    @validate_params(check_rules).to_return(False)
    def basic_func(a: int, b: dict):
        return f"{a} is an int and {b} is a dict"

    res = basic_func(a_value, b_value)
    assert res == value
