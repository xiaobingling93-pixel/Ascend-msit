from typing import Union
import components
from components.utils.check.checker import Checker, CheckResult, rule


class NumberChecker(Checker):

    @rule()
    def is_number(self) -> Union["NumberChecker", CheckResult]:
        return CheckResult(isinstance(self.instance, (int, float)), "not number")

    @rule()
    def is_int(self) -> Union["NumberChecker", CheckResult]:
        return CheckResult(isinstance(self.instance, int), "not integer")

    @rule()
    def is_float(self) -> Union["NumberChecker", CheckResult]:
        return CheckResult(isinstance(self.instance, float), "not float")

    @rule()
    def is_zero(self) -> Union["NumberChecker", CheckResult]:
        return CheckResult(self.instance == 0, "not zero")

    @rule()
    def is_divisible_by(self, target) -> Union["NumberChecker", CheckResult]:
        return CheckResult(target % self.instance == 0, f"not divisible by {target}")

    @rule()
    def in_range(self, min_value, max_value) -> Union["NumberChecker", CheckResult]:
        return self.min(min_value) and self.max(max_value)

    @rule()
    def min(self, min_value) -> Union["NumberChecker", CheckResult]:
        return CheckResult(min_value <= self.instance, f"less than {min_value}")

    @rule()
    def max(self, max_value) -> Union["NumberChecker", CheckResult]:
        return CheckResult(self.instance <= max_value, f"more than {max_value}")

    @rule()
    def less_than(self, max_value) -> Union["NumberChecker", CheckResult]:
        return CheckResult(max_value > self.instance, f"not less than {max_value}")

    @rule()
    def more_than(self, min_value) -> Union["NumberChecker", CheckResult]:
        return CheckResult(self.instance > min_value, f"not more than {min_value}")
