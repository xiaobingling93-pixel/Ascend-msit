# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

from components.utils.check import Rule, ObjectChecker


class TmpObj:
    def __init__(self, param_a, param_b) -> None:
        self.p_a = param_a
        self.p_b = param_b


def test_obj_checker_given_attr_rule_when_any_pass():
    obj_rule = ObjectChecker().is_attrs_valid(p_a=Rule.num(), p_b=Rule.str())
    assert obj_rule.check(TmpObj(1, "1"))


def test_obj_checker_given_attr_rule_when_any_failed():
    obj_rule = ObjectChecker().is_attrs_valid(p_a=Rule.num(), p_b=Rule.str())
    assert not obj_rule.check(TmpObj("1", 1))
    assert len(str(obj_rule.check(TmpObj("1", 1))).split("\n")) == 2
