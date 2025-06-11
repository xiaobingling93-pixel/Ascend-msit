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
import sys
import traceback

import functools
from abc import abstractmethod
from packaging.version import Version

from .hook_helper import HookHelper


class VLLMHookerBase:
    @staticmethod
    def get_parents_name(ori_func, index=1):
        gen = traceback.walk_stack(None)
        try:
            for _ in range(index + 1):
                f = next(gen)
            return f[0].f_code.co_name
        except StopIteration:
            return None

    @abstractmethod
    def init(self):
        pass

    def do_hook(self, hook_points, profiler_func_maker, pname=None):
        for ori_func in hook_points:
            profiler_func = profiler_func_maker(ori_func)

            def replace_func(ori_func, pname):
                @functools.wraps(ori_func)
                def wrapper(*args, **kwargs):
                    if pname is not None and self.get_parents_name(ori_func) != pname:
                        return ori_func(*args, **kwargs)
                    return profiler_func(*args, **kwargs)

                return wrapper

            HookHelper(ori_func, replace_func(ori_func, pname)).replace()

    def support_version(self, version):
        if hasattr(self, "vllm_version"):
            min_version = self.vllm_version[0]
            max_version = self.vllm_version[1]
            if min_version is not None and Version(min_version) > Version(version):
                return False
            if max_version is not None and Version(max_version) < Version(version):
                return False
        return True
