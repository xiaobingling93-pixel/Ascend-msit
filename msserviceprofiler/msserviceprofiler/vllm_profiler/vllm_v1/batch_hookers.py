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
from ..utils import logger


def compare_deques(queue1, queue2):
    counter1 = Counter(queue1)
    counter2 = Counter(queue2)
    diff = counter1 - counter2
    return diff


def queue_profiler(before_queue, after_queue, queue_name):
    # 队列元素减少
    less_queue = compare_deques(before_queue, after_queue)
    rid_list = []
    for seq_group in less_queue: # V1 note: SequenceGroup == Request
        rid_list.append(seq_group.request_id)
    if len(rid_list) > 0:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(rid_list)
        prof.metric("QueueSize", len(after_queue)).metric_scope("QueueName", queue_name).event("Dequeue")

    # 队列元素增加
    add_queue = compare_deques(after_queue, before_queue)
    rid_list.clear()
    for seq_group in add_queue: # V1 note: SequenceGroup == Request
        rid_list.append(seq_group.request_id)
    if len(rid_list) > 0:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(rid_list)
        prof.metric("QueueSize", len(after_queue)).metric_scope("QueueName", queue_name).event("Enqueue")


class HookState:
    def __init__(self):
        self.request_id_to_prompt_token_len = {}
        self.request_id_to_iter_size = {}
        self.running = set()
        self.waiting = set()


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
    Profiler(Level.INFO).domain("BatchSchedule").res(request_id).event("ReqState")
    return ret


@vllm_hook(("vllm.v1.core.sched.scheduler", "Scheduler.schedule"), min_version="0.9.1")
def schedule(original_func, this, *args, **kwargs):
    state = _get_state()
    prof = Profiler(Level.INFO).domain("BatchSchedule").span_start("batchFrameworkProcessing")

    before_running_queue = this.running
    before_waiting_queue = this.waiting
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
    
    if not request_id_list:
        return scheduler_output

    queue_profiler(before_running_queue, this.running, "running")
    queue_profiler(before_waiting_queue, this.waiting, "waiting")
    prof.res(request_id_with_iter_list)
    
    # 新的请求从WAITING -> RUNNING
    for scheduled_new_req in scheduler_output.scheduled_new_reqs:
        queue_prof = Profiler(Level.INFO).domain("BatchSchedule").res(scheduled_new_req.req_id)
        state.running.add(scheduled_new_req.req_id)
        state.waiting.remove(scheduled_new_req.req_id)
        logger.debug(f">>> [queue-waiting][WAITING -> RUNNING]: {len(state.waiting)}, [queue-running]: {len(state.running)}")
        queue_prof.metric_inc("RUNNING", 1).metric_inc("WAITING", -1).event("ReqState")
    
    # PREEMPTED请求从WAITING -> RUNNING
    for scheduled_cached_req in scheduler_output.scheduled_cached_reqs:
        if scheduled_cached_req.req_id not in state.running and scheduled_cached_req.req_id in state.waiting:
            queue_prof = Profiler(Level.INFO).domain("BatchSchedule").res(scheduled_cached_req.req_id)
            state.running.add(scheduled_new_req.req_id)
            state.waiting.remove(scheduled_new_req.req_id)
            logger.debug(f">>> [queue-waiting][PREEMPTED -> RUNNING]: {len(state.waiting)}, [queue-running]: {len(state.running)}")
            queue_prof.metric_inc("RUNNING", 1).metric_inc("WAITING", -1).event("ReqState")
    
    # running的请求被抢占从RUNNING -> WAITING
    for request_id in state.running:
        if request_id in this.waiting:
            queue_prof = Profiler(Level.INFO).domain("BatchSchedule").res(request_id)
            state.waiting.add(scheduled_new_req.req_id)
            state.running.remove(scheduled_new_req.req_id)
            logger.debug(f">>> [queue-waiting][RUNNING -> WAITING]: {len(state.waiting)}, [queue-running]: {len(state.running)}")
            queue_prof.metric_inc("RUNNING", -1).metric_inc("WAITING", 1).event("ReqState") 
    
    logger.debug(f" state.request_id_to_iter_size: {state.request_id_to_iter_size}")
    is_prefill = any(val == 0 for val in state.request_id_to_iter_size.values())
    # TODO prefill的判断逻辑需要根据整个batch来看是prefill还是decode还是mix
    prof.attr("batch_type", "Prefill" if is_prefill else "Decode")
    prof.span_end()

    return scheduler_output


@vllm_hook(("vllm.v1.core.sched.scheduler", "Scheduler._free_request"), min_version="0.9.1")
def free_request(original_func, this, request, *args, **kwargs):
    original_func(this, request, *args, **kwargs)
    state = _get_state()
    prof = Profiler(Level.INFO).domain("BatchSchedule").res(request.request_id)
    if request.request_id in state.running:
        prof.metric_inc("RUNNING", -1).metric_inc("FINISHED", 1).event("ReqState")
        state.running.remove(request.request_id)
    elif request.request_id in state.waiting:
        prof.metric_inc("WAITING", -1).metric_inc("FINISHED", 1).event("ReqState")
        state.waiting.remove(request.request_id)
    logger.debug(f">>> [queue-waiting][free_request]: {len(state.waiting)}, [queue-running]: {len(state.running)}")


@vllm_hook(("vllm.v1.core.sched.scheduler", "Scheduler.add_request"), min_version="0.9.1")
def add_request(original_func, this, request, *args, **kwargs):
    original_func(this, request, *args, **kwargs)
    state = _get_state()
    state.waiting.add(request.request_id)
    logger.debug(f">>> [queue-waiting][add_request]: {len(state.waiting)}, [queue-running]: {len(state.running)}")
    prof = Profiler(Level.INFO).domain("BatchSchedule").res(request.request_id)
    prof.metric_inc("WAITING", 1).event("ReqState")
    prof_queue = Profiler(Level.INFO).domain("BatchSchedule").res(request.request_id)
    prof_queue.metric("QueueSize", len(this.waiting)).metric_scope("queue_type", "WAITING").event("Enqueue")
