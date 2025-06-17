# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import os
import logging
from enum import Enum
from collections import namedtuple

from msprechecker.core.utils.perm import FilePerm

# 创建一个全局的注册表，注册为分析函数
REGISTRY = {}

ANSWERS = {}
CONTENT_PARTS = namedtuple("CONTENT_PARTS", ["before", "after", "sys"])("before", "after", "sys")
CONTENTS = {}  # Will save to file in the end


def register_checker(analyze_name=None):
    def decorator(func):
        name = analyze_name if analyze_name is not None else func.__name__
        REGISTRY[name] = func
        return func

    return decorator


def cached():
    # 缓存函数结果，反正所有输入都是一样的
    cache = {}

    def decorator(func):
        name = func.__name__

        def wrapper(*args, **kwargs):
            if name in cache:
                return cache[name]
            result = func(*args, **kwargs)
            cache[name] = result
            return result

        return wrapper

    return decorator


def answer(suggesion_type=None, suggesion_item=None, action=None, reason=""):
    ANSWERS[suggesion_type].setdefault(suggesion_item, []).append((action, reason))


def record(content, part=CONTENT_PARTS.after):
    CONTENTS.setdefault(part, []).append(content)


CheckResult = Enum("CheckResult", ["OK", "UNFINISH", "WARN", "ERROR", "VIP"])


def show_check_result(domain, checker, result=None, action=None, reason=None):
    color_and_text = {
        CheckResult.OK: ("\033[92m", "OK"),
        CheckResult.UNFINISH: ("\033[93m", "UNFINISH"),
        CheckResult.WARN: ("\033[93m", "WARN"),
        CheckResult.ERROR: ("\033[91m", "NOK"),
        CheckResult.VIP: ("\033[94m", action),
    }

    if result is None:
        color, text = "\033[97m", ""
    else:
        color, text = color_and_text.get(result, ("\033[97m", ""))
    print(f"- {domain} {color}[{text}]\033[0m {checker} ")
    if action is not None and result != CheckResult.VIP:
        print(f"    * {action}")
    if reason is not None:
        print(f"    * {reason}")


class PrecheckerBase:
    __checker_name__ = "undefined"

    def __init__(self):
        self.logger = self._init_logger()

    @staticmethod
    def _init_logger():
        local_logger = logging.getLogger(__name__)
        local_logger.setLevel(logging.INFO)
        local_logger.propagate = False

        if not local_logger.handlers:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            stream_handler.setFormatter(formatter)
            local_logger.addHandler(stream_handler)
        return local_logger

    @classmethod
    def name(cls):
        return cls.__checker_name__

    def do_precheck(self, envs, **kwargs):
        pass

    def collect_env(self, **kwargs):
        return None

    def precheck(self, **kwargs):
        envs = self.collect_env(**kwargs)
        self.do_precheck(envs, **kwargs)

    def show_check_result(self, checker, result=None, action=None, reason=None):
        show_check_result(self.name(), checker, result, action, reason)


class GroupPrechecker(PrecheckerBase):
    __checker_name__ = "undefined"

    def __init__(self, *sub):
        super().__init__()
        self.sub_checkers = self.init_sub_checkers()

    def init_sub_checkers(self):
        return []

    def collect_env(self, **kwargs):
        group_envs = {}
        for sub in self.sub_checkers:
            group_envs.update({sub.name(): sub.collect_env(**kwargs)})
        return group_envs

    def do_precheck(self, envs, **kwargs):
        for sub in self.sub_checkers:
            sub.do_precheck(envs.get(sub.name()))


def check_file_permission(filepath, domain="config", checker_name="file_perm"):
    try:
        perm = FilePerm(oct(os.stat(filepath).st_mode & 0o777))
        if perm > FilePerm(0o640):
            show_check_result(
                domain,
                checker_name,
                CheckResult.ERROR,
                action=f"请修改 {filepath} 权限为小于 640 (如 chmod 640 {filepath})",
                reason=f"当前权限为 {perm}，出于安全考虑，配置文件的权限应该不能超过 0o640",
            )
            return False
    except Exception as e:
        show_check_result(
            domain,
            checker_name,
            CheckResult.ERROR,
            action=f"无法检查 {filepath} 权限",
            reason=str(e),
        )
        return False
    return True
