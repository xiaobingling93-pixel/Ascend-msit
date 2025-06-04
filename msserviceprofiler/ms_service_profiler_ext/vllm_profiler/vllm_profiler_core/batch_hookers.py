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
from collections import Counter
from ms_service_profiler import Profiler, Level
from .vllm_hooker_base import VLLMHookerBase


def compare_deques(queue1, queue2):
    counter1 = Counter(queue1)
    counter2 = Counter(queue2)
    diff = counter1 - counter2
    return diff


def queue_profiler(before_queue, after_queue, queue_name):
    # 队列元素减少
    less_queue = compare_deques(before_queue, after_queue)
    rid_list = []
    for seq_group in less_queue:
        rid_list.append(seq_group.request_id)
    if len(rid_list) > 0:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(rid_list)
        prof.metric("QueueSize", len(after_queue)).metric_scope(queue_name).event("Dequeue")

    # 队列元素增加
    add_queue = compare_deques(after_queue, before_queue)
    rid_list.clear()
    for seq_group in add_queue:
        rid_list.append(seq_group.request_id)
    if len(rid_list) > 0:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(rid_list)
        prof.metric("QueueSize", len(after_queue)).metric_scope(queue_name).event("Enqueue")


class SchedulerHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.8.4")

    def init(self):
        from vllm.core.scheduler import Scheduler
        from vllm.sequence import SequenceGroupMetadata, SequenceStatus

        def schedule_maker(ori_func):
            def schedule(this, *args, **kwargs):
                prof = Profiler(Level.INFO).domain("BatchSchedule").span_start("batchFrameworkProcessing")
                seq_group_metadata_list, scheduler_outputs, allow_async_output_proc = ori_func(this, *args, **kwargs)

                iter_size_map, is_prefill = {}, False
                for seq_group in this.running:
                    if len(seq_group.seqs) == 0:
                        continue
                    prompt_len = len(seq_group.prompt_token_ids)
                    generated_len = len(seq_group.seqs[0].get_token_ids()) - prompt_len
                    iter_size_map[seq_group.request_id] = generated_len
                    is_prefill = is_prefill or (generated_len == 0)
                prof.attr("batch_type", "Prefill" if is_prefill else "Decode")

                rid_list = []
                for metadata in seq_group_metadata_list:
                    if isinstance(metadata, SequenceGroupMetadata):
                        iter_size = iter_size_map.get(metadata.request_id, metadata.token_chunk_size)
                        data = {"rid": metadata.request_id, "iter_size": iter_size}
                        rid_list.append(data)
                prof.res(rid_list)
                prof.span_end()

                return seq_group_metadata_list, scheduler_outputs, allow_async_output_proc

            return schedule

        self.do_hook([Scheduler.schedule], schedule_maker)

        def abort_seq_group_maker(ori_func):
            def abort_seq_group(this, request_id, *args, **kwargs):
                prof = Profiler(Level.INFO).domain("BatchSchedule").res(request_id)
                prof.metric_inc("FINISHED_ABORTED", 1).event("ReqState")
                ori_func(this, request_id, *args, **kwargs)

            return abort_seq_group

        self.do_hook([Scheduler.abort_seq_group], abort_seq_group_maker)

        def allocate_and_set_running_maker(ori_func):
            def _allocate_and_set_running(this, seq_group, *args, **kwargs):
                prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
                prof.metric_inc("RUNNING", 1).metric_inc("WAITING", -1).event("ReqState")
                ori_func(this, seq_group, *args, **kwargs)

            return _allocate_and_set_running

        self.do_hook([Scheduler._allocate_and_set_running], allocate_and_set_running_maker)

        def preempt_by_recompute_maker(ori_func):
            def _preempt_by_recompute(this, seq_group, *args, **kwargs):
                seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
                if len(seqs) != 1:
                    prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
                    prof.metric_inc("RUNNING", -1).metric_inc("WAITING", 1).event("ReqState")
                ori_func(this, seq_group, *args, **kwargs)

            return _preempt_by_recompute

        self.do_hook([Scheduler._preempt_by_recompute], preempt_by_recompute_maker)

        def swap_in_maker(ori_func):
            def _swap_in(this, seq_group, *args, **kwargs):
                prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
                prof.metric_inc("RUNNING", 1).metric_inc("SWAPPED", -1).event("ReqState")
                ori_func(this, seq_group, *args, **kwargs)

            return _swap_in

        self.do_hook([Scheduler._swap_in], swap_in_maker)

        def swap_out_maker(ori_func):
            def _swap_out(this, seq_group, *args, **kwargs):
                if this.block_manager.can_swap_out(seq_group):
                    prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
                    prof.metric_inc("RUNNING", -1).metric_inc("SWAPPED", 1).event("ReqState")
                ori_func(this, seq_group, *args, **kwargs)

            return _swap_out

        self.do_hook([Scheduler._swap_out], swap_out_maker)

        def free_finished_seq_groups_maker(ori_func):
            def free_finished_seq_groups(this, *args, **kwargs):
                before_running_queue = this.running
                for seq_group in this.running:
                    prof = Profiler(Level.INFO).domain("BatchSchedule").res(seq_group.request_id)
                    prof.metric_inc("RUNNING", -1).metric_inc("FINISHED", 1).event("ReqState")
                ori_func(this, *args, **kwargs)
                queue_profiler(before_running_queue, this.running, "running")

            return free_finished_seq_groups

        self.do_hook([Scheduler.free_finished_seq_groups], free_finished_seq_groups_maker)

        def add_seq_group_to_running_maker(ori_func):
            def _add_seq_group_to_running(this, seq_group, *args, **kwargs):
                ori_func(this, seq_group, *args, **kwargs)
                prof = Profiler(Level.INFO).domain("BatchSchedule").res([seq_group.request_id])
                prof.metric("QueueSize", len(this.running)).metric_scope("running").event("Enqueue")

            return _add_seq_group_to_running

        self.do_hook([Scheduler._add_seq_group_to_running], add_seq_group_to_running_maker)

        def schedule_priority_preemption_maker(ori_func):
            def _schedule_priority_preemption(this, budget, *args, **kwargs):
                before_waiting_queue = this.waiting
                before_running_queue = this.running
                force_preemption_count = ori_func(this, budget, *args, **kwargs)
                queue_profiler(before_waiting_queue, this.waiting, "waiting")
                queue_profiler(before_running_queue, this.running, "running")
                return force_preemption_count

            return _schedule_priority_preemption

        self.do_hook([Scheduler._schedule_priority_preemption], schedule_priority_preemption_maker)

        def schedule_default_maker(ori_func):
            def _schedule_default(this, *args, **kwargs):
                before_swapped_queue = this.swapped
                before_running_queue = this.running
                before_waiting_queue = this.waiting
                scheduler_outputs = ori_func(this, *args, **kwargs)
                queue_profiler(before_swapped_queue, this.swapped, "swapped")
                queue_profiler(before_running_queue, this.running, "running")
                queue_profiler(before_waiting_queue, this.waiting, "waiting")
                return scheduler_outputs

            return _schedule_default

        self.do_hook([Scheduler._schedule_default], schedule_default_maker)

        def schedule_chunked_prefill_maker(ori_func):
            def _schedule_chunked_prefill(this, *args, **kwargs):
                before_running_queue = this.running
                before_waiting_queue = this.waiting
                before_swapped_queue = this.swapped
                scheduler_outputs = ori_func(this, *args, **kwargs)
                queue_profiler(before_running_queue, this.running, "running")
                queue_profiler(before_waiting_queue, this.waiting, "waiting")
                queue_profiler(before_swapped_queue, this.swapped, "swapped")
                return scheduler_outputs

            return _schedule_chunked_prefill

        self.do_hook([Scheduler._schedule_chunked_prefill], schedule_chunked_prefill_maker)

        def add_seq_group_maker(ori_func):
            def add_seq_group(this, seq_group, *args, **kwargs):
                ori_func(this, seq_group, *args, **kwargs)
                prof = Profiler(Level.INFO).domain("BatchSchedule").res([seq_group.request_id])
                prof.metric("QueueSize", len(this.waiting)).metric_scope("waiting").event("Enqueue")

            return add_seq_group

        self.do_hook([Scheduler.add_seq_group], add_seq_group_maker)

        def add_seq_group_to_swapped_maker(ori_func):
            def _add_seq_group_to_swapped(this, seq_group, *args, **kwargs):
                ori_func(this, seq_group, *args, **kwargs)
                prof = Profiler(Level.INFO).domain("BatchSchedule").res([seq_group.request_id])
                prof.metric("QueueSize", len(this.swapped)).metric_scope("swapped").event("Enqueue")

            return _add_seq_group_to_swapped

        self.do_hook([Scheduler._add_seq_group_to_swapped], add_seq_group_to_swapped_maker)


class LLMEngineHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.8.4")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine

        def add_processed_request_maker(ori_func):
            def _add_processed_request(this, request_id, *args, **kwargs):
                ori_func(this, request_id, *args, **kwargs)
                Profiler(Level.INFO).domain("Request").res(request_id).metric_inc("WAITING", 1).event("ReqState")

            return _add_processed_request

        self.do_hook([getattr(LLMEngine, "_add_processed_request")], add_processed_request_maker)


batch_hookers = [SchedulerHook, LLMEngineHook]
