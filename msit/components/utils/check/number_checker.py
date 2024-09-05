from typing import Union
import components
from components.utils.check.checker import Checker, CheckResult, rule


class NumberChecker(Checker):

    @rule()
    def is_number(self) -> Union["NumberChecker", CheckResult]:
        return CheckResult(isinstance(self.instance, (int, float)), "not number")

    @rule()
    def is_int(self) -> Union["NumberChecker", CheckResult]:
        return CheckResult(isinstance(self.instance, int) and not isinstance(self.instance, bool), "not integer")

    @rule()
    def is_float(self) -> Union["NumberChecker", CheckResult]:
        return CheckResult(isinstance(self.instance, float), "not float")

    @rule()
    def is_zero(self) -> Union["NumberChecker", CheckResult]:
        return CheckResult(self.instance == 0, "not zero")

    @rule()
    def is_divisible_by(self, target) -> Union["NumberChecker", CheckResult]:
        if self.instance != 0:
            return CheckResult(target % self.instance == 0, f"not divisible by {target}")
        else:
            raise ValueError(f"Failed to check division, since {self.instance} is 0.")

    @rule()
    def in_range(self, min_value, max_value) -> Union["NumberChecker", CheckResult]:
        return self.greater_equal(min_value) and self.less_equal(max_value)

    @rule()
    def greater_equal(self, value) -> Union["NumberChecker", CheckResult]:
        return CheckResult(value <= self.instance, f"not greater equal to {value}")

    @rule()
    def less_equal(self, value) -> Union["NumberChecker", CheckResult]:
        return CheckResult(self.instance <= value, f"not less equal to {value}")

    @rule()
    def less_than(self, value) -> Union["NumberChecker", CheckResult]:
        return CheckResult(value > self.instance, f"not less than {value}")

    @rule()
    def greater_than(self, value) -> Union["NumberChecker", CheckResult]:
        return CheckResult(self.instance > value, f"not greater than {value}")
