from vllm_profiler.vllm_profiler_core.vllm_hookers import all_hookers

for hook_cls in all_hookers:
    hooker = hook_cls()
    if hooker.support_version("0.6.3"):
        hooker.init()
