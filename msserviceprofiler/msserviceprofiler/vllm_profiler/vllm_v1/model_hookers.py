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
class HookState:
    def __init__(self):
        self.forward_profiler = None
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


def _extract_request_id_from_scheduler_output(scheduler_output, state):
    request_id_list, request_id_with_iter_list = [], []
    for request_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
        request_id_list.append({"rid": request_id})
        iter_size = state.request_id_to_iter_size.get(request_id, -1) + 1  # start from 0
        request_id_with_iter_list.append({"rid": request_id, "iter_size": iter_size})
        state.request_id_to_iter_size[request_id] = iter_size

        if request_id in scheduler_output.finished_req_ids:
            state.request_id_to_prompt_token_len.pop(request_id, None)
            state.request_id_to_iter_size.pop(request_id, None)
    return request_id_list,request_id_with_iter_list


@vllm_hook(hook_points=("vllm_ascend.worker.model_runner_v1", "NPUModelRunner._update_states"), min_version="0.9.1")
def update_states(original_func, this, scheduler_output, *args, **kwargs):
    """处理执行模型钩子"""
    state = _get_state()
    request_id_list, request_id_with_iter_list = _extract_request_id_from_scheduler_output(scheduler_output, state)
    prof = Profiler(Level.INFO).domain("ModelExecute").res(request_id_with_iter_list)
    prof.span_start("processRequestState")
    original_func(this, scheduler_output, *args, **kwargs)
    prof.span_end()


@vllm_hook(hook_points=("vllm.model_executor.layers.logits_processpr", "LogitsProcessor.forward"), min_version="0.9.1")
def compute_logits(original_func, this, *args, **kwargs):
    """处理执行模型钩子"""
    prof = Profiler(Level.INFO).domain("ModelExecute").span_start("computing_logits")
    original_func(this, *args, **kwargs)
    prof.span_end()


@vllm_hook(hook_points=("vllm.v1.sampler", "Sampler.forward"), min_version="0.9.1")
def sampler_forward(original_func, this, *args, **kwargs):
    """处理执行模型钩子"""
    prof = Profiler(Level.INFO).domain("ModelExecute").span_start("sampling")
    original_func(this, *args, **kwargs)
    prof.span_end()


@vllm_hook(hook_points=("vllm.v1.executor.abstract", "Executor.execute_model"), min_version="0.9.1")
def execute_model(original_func, this, scheduler_output, *args, **kwargs):
    """处理执行模型钩子"""
    state = _get_state()
    for scheduled_new_req in scheduler_output.scheduled_new_reqs:
        state.request_id_to_prompt_token_len[scheduled_new_req.req_id] = len(scheduled_new_req.prompt_token_ids)

    request_id_list, request_id_with_iter_list = _extract_request_id_from_scheduler_output(scheduler_output, state)

    is_prefill = False
    for scheduled_req in scheduler_output.scheduled_cached_reqs + scheduler_output.scheduled_new_reqs:
        request_id = scheduled_req.req_id
        if request_id not in state.request_id_to_prompt_token_len:
            continue
        is_prefill |= (scheduled_req.num_computed_tokens < state.request_id_to_prompt_token_len[request_id])

    if request_id_list:
        prof = Profiler(Level.INFO).domain("ModelExecute")
        prof.res(request_id_with_iter_list)
        prof.attr("batch_type", "Prefill" if is_prefill else "Decode")  # [TODO] for v1, prefill is combined with decode
        prof.span_start("modelExec")
        prof.attr("batch_size", scheduler_output.total_num_scheduled_tokens)

        state.forward_profiler = Profiler(Level.INFO).domain("ModelExecute").res(request_id_list)

    ret = original_func(this, scheduler_output, *args, **kwargs)
    if request_id_list:
        prof.span_end()
    return ret


@vllm_hook(("vllm.forward_context", "set_forward_context"), min_version="0.9.1")
@contextmanager
def set_forward_context(original_func, *args, **kwargs):
    """前向上下文钩子"""
    state = _get_state()
    if state.forward_profiler is not None:
        state.forward_profiler.span_start("forward")

    with original_func(*args, **kwargs):
        yield
    if state.forward_profiler is not None:
        state.forward_profiler.span_end()
        state.forward_profiler = None
