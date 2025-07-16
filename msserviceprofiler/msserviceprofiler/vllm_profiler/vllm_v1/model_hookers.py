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
import threading
from contextlib import contextmanager
from ms_service_profiler import Profiler, Level
from ..module_hook import vllm_hook


# 线程安全的全局状态
class HookState:cp ms_service_profiler
    def __init__(self):
        self.forward_profiler = []
        self.execute_model_first_run = True
        self.begin_forward_first_run = True
        self.request_id_to_prompt_token_len = {}
        self.request_id_to_iter_size = {}


# 线程本地存储
_thread_local = threading.local()


def _get_state() -> HookState:
    """获取线程本地状态"""
    if not hasattr(_thread_local, "hook_state"):
        _thread_local.hook_state = HookState()
    return _thread_local.hook_state


@vllm_hook(hook_points=("vllm.v1.executor.abstract", "Executor.execute_model"), min_version="0.9.1")
def handle_execute_model(original_func, this, scheduler_output, *args, **kwargs):
    """处理执行模型钩子"""
    state = _get_state()
    for scheduled_new_req in scheduler_output.scheduled_new_reqs:
        state.request_id_prompt_token_len[scheduled_new_req.req_id] = len(scheduled_new_req.prompt_token_ids)

    prof = Profiler(Level.INFO).domain("ModelExecute")
    request_id_list, request_id_with_iter_list = [], []
    for request_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
        request_id_list.append({"rid": request_id})
        iter_size = state.request_id_to_iter_size.get(request_id, -1) + 1  # start from 0
        request_id_with_iter_list.append({"rid": seq_metadata.request_id, "iter_size": iter_size})
        state.request_id_to_iter_size[request_id] = iter_size

    if request_id in scheduler_output.finished_request_ids:
        state.request_id_prompt_token_len.pop(request_id, None)
        state.request_id_to_iter_size.pop(request_id, None)

    prof.res(request_id_with_iter_list)
    # prof.attr("batch_type", "Prefill" if is_prefill else "Decode")  # [TODO] for v1, prefill is combined with decode
    prof.span_start("modelExec")
    prof.attr("batch_size", scheduler_output.total_num_scheduled_tokens)

    preprocess_prof = Profiler(Level.INFO).domain("ModelExecute").res(request_id_list)
    preprocess_prof.event("preprocess")
    forward_prof = Profiler(Level.INFO).domain("ModelExecute").res(request_id_list)
    state.forward_profiler.append(forward_prof)

    ret = original_func(this, execute_model_req, *args, **kwargs)
    prof.span_end()
    return ret


@vllm_hook(("vllm.forward_context", "set_forward_context"), min_version="0.8.4")
@contextmanager
def set_forward_context(original_func, *args, **kwargs):
    """前向上下文钩子"""
    state = _get_state()
    if len(state.forward_profiler) > 0:
        prof = state.forward_profiler.pop(0)
        prof.span_start("forward")
    else:
        prof = None
    with original_func(*args, **kwargs):
        yield
    if prof is not None:
        prof.span_end()
