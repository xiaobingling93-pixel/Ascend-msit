import pytest

from components.utils.check.list_checker import ListChecker
from components.utils.check.number_checker import NumberChecker


@pytest.mark.parametrize("param, value", [
    ([1, [2, 3]], True),
    ([], True),
    (1, False),
])
def test_is_list(param, value):
    list_checker = ListChecker(param)
    assert list_checker.is_list().passed is value


@pytest.mark.parametrize("param, value", [
    ([1, 2, 3], True),
    ([], False),
])
def test_is_list_not_empty(param, value):
    list_checker = ListChecker(param)
    assert list_checker.is_list_not_empty().passed is value


@pytest.mark.parametrize("param, check_rule, value", [
    ([1, 2, 3], NumberChecker().is_int(), True),
    (["a", 2, "c"], NumberChecker().is_int(), False),
])
def test_is_element_valid(param, check_rule, value):
    list_checker = ListChecker(param)
    assert list_checker.is_element_valid(check_rule).passed is value


@pytest.mark.parametrize("param, min_length, max_length, value", [
    ([1, 2, 3], 3, None, True),
    ([1, 2, 3], None, 3, True),
    ([1, 2, 3], 1, 4, True),
    ([1, 2, 3], 2, 2, False),
])
def test_is_length_valid(param, min_length, max_length, value):
    list_checker = ListChecker(param)
    assert list_checker.is_length_valid(min_length, max_length).passed is value
