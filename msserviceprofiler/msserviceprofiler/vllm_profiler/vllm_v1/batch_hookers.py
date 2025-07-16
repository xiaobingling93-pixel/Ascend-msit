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
from collections import Counter
from ms_service_profiler import Profiler, Level
from ..module_hook import vllm_hook

class HookState:
    def __init__(self):
        self.request_id_to_prompt_token_len = {}
        self.request_id_to_iter_size = {}


# 线程本地存储
_thread_local = threading.local()


def _get_state() -> HookState:
    """获取线程本地状态"""
    if not hasattr(_thread_local, "hook_state"):
        _thread_local.hook_state = HookState()
    return _thread_local.hook_state


@vllm_hook(("vllm.v1.engine.processor", "Processor.process_inputs"), min_version="0.9.1")
def process_inputs(original_func, this, request_id, *args, **kwargs):
    ret = original_func(this, request_id, *args, **kwargs)
    Profiler(Level.INFO).domain("BatchSchedule").res(request_id).metric_inc("WAITING", 1).event("ReqState")
    return ret

@vllm_hook(("vllm.v1.core.sched.scheduler", "Scheduler.schedule"), min_version="0.9.1")
def schedule(original_func, this, *args, **kwargs):
    # from vllm.sequence import SequenceGroupMetadata
    state = _get_state()
    prof = Profiler(Level.INFO).domain("BatchSchedule").span_start("batchFrameworkProcessing")
    scheduler_output = original_func(this, *args, **kwargs)

    for scheduled_new_req in scheduler_output.scheduled_new_reqs:
        state.request_id_to_prompt_token_len[scheduled_new_req.req_id] = len(scheduled_new_req.prompt_token_ids)

    request_id_list, request_id_with_iter_list = [], []
    for request_id, _ in scheduler_output.num_scheduled_tokens.items():
        request_id_list.append({"rid": request_id})
        iter_size = state.request_id_to_iter_size.get(request_id, -1) + 1  # start from 0
        request_id_with_iter_list.append({"rid": request_id, "iter_size": iter_size})
        state.request_id_to_iter_size[request_id] = iter_size

        if request_id in scheduler_output.finished_req_ids:
            state.request_id_to_prompt_token_len.pop(request_id, None)
            state.request_id_to_iter_size.pop(request_id, None)
        
    is_prefill = any(state.request_id_to_iter_size.values() == 0)
    prof.attr("batch_type", "Prefill" if is_prefill else "Decode")

    prof.res(request_id_with_iter_list)
    prof.span_end()

    return scheduler_output
