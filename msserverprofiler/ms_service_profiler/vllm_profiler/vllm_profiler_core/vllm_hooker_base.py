import traceback

from packaging.version import Version
from abc import abstractmethod
from .hook_helper import HookHelper
import functools
import sys


class vLLMHookerBase:

    def do_hook(self, hook_points, profiler_func_maker, pname=None):
        for ori_func in hook_points:
            profiler_func = profiler_func_maker(ori_func)

            @functools.wraps(ori_func)
            def replace_func(*args, **kwargs):
                if pname is not None and self.get_parents_name(ori_func) != pname:
                    return ori_func(*args, **kwargs)
                return profiler_func(*args, **kwargs)

            HookHelper(ori_func, replace_func).replace()

    @abstractmethod
    def init(self):
        pass

    def get_parents_name(self, ori_func, index=1):
        gen = traceback.walk_stack(None)
        try:
            for _ in range(index + 1):
                f = next(gen)
            print("f:", f[0])
            return f[0].f_code.co_name
        except StopIteration:
            return None

    def support_version(self, version):
        if hasattr(self, "vllm_version"):
            min_version = self.vllm_version[0]
            max_version = self.vllm_version[1]
            if min_version is not None and Version(min_version) > Version(version):
                return False
            if max_version is not None and Version(max_version) < Version(version):
                return False

        return True
