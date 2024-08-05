from typing import Union

from components.utils.check.checker import Checker, CheckResult, rule
from components.utils.check.number_checker import NumberChecker


class ListChecker(Checker):

    @rule()
    def is_list(self) -> Union["ListChecker", CheckResult]:
        return isinstance(self.instance, list), "Input object is not a list."

    @rule()
    def is_list_not_empty(self) -> Union["ListChecker", CheckResult]:
        return bool(self.instance), "List is empty."

    @rule()
    def is_element_valid(self, check_rule) -> Union["ListChecker", CheckResult]:
        is_pass = self.is_list().passed
        if not is_pass:
            return is_pass, "Input object is not a list."
        err_msgs = []
        for element in self.instance:
            rule_is_pass = check_rule.check(element)
            if not rule_is_pass:
                is_pass = False
                err_msgs.append(f"{element}: {str(rule_is_pass)}")
        return is_pass, "\n".join(err_msgs)

    @rule()
    def is_length_valid(self, min_length=None, max_length=None) -> Union["ListChecker", CheckResult]:
        length = len(self.instance)
        length_valid = True
        err_msg = []
        if min_length is not None and length < min_length:
            length_valid = False
            err_msg.append(f"List is shorter than the minimum length of {min_length}.")
        if max_length is not None and length > max_length:
            length_valid = False
            err_msg.append(f"List is longer than the maximum length of {max_length}.")
        return length_valid, "\n".join(err_msg)
