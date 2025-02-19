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
from ms_service_profiler import Profiler, Level
from vllm.sequence import SequenceGroupMetadata, SequenceStatus
from .vllm_hooker_base import VLLMHookerBase


class SchedulerHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.core.scheduler import Scheduler

        def schedule_maker(ori_func):
            def schedule(this, *args, **kwargs):
                prof = Profiler(Level.INFO).span_start("BatchSchedule")
                seq_group_metadata_list, scheduler_outputs, allow_async_output_proc = ori_func(this, *args, **kwargs)

                rid_list = []
                for seq_group_metadata in seq_group_metadata_list:
                    if isinstance(seq_group_metadata, SequenceGroupMetadata):
                        data = {"rid": seq_group_metadata.request_id, "iter": seq_group_metadata.token_chunk_size}
                        rid_list.append(data)
                prof.res(rid_list)
                prof.span_end()

                return seq_group_metadata_list, scheduler_outputs, allow_async_output_proc

            return schedule

        self.do_hook([Scheduler.schedule], schedule_maker)

        def abort_seq_group_maker(ori_func):
            def abort_seq_group(this, request_id, *args, **kwargs):
                Profiler(Level.INFO).res(request_id).metric_inc('FINISHED_ABORTED', 1).event("ReqState")
                ori_func(this, request_id, *args, **kwargs)
            return abort_seq_group
        
        self.do_hook([Scheduler.abort_seq_group], abort_seq_group_maker)

        def allocate_and_set_running_maker(ori_func):
            def _allocate_and_set_running(this, seq_group, *args, **kwargs):
                Profiler(Level.INFO).res(seq_group.request_id).metric_inc('RUNNING', 1).\
                    metric_inc('WAITING', -1).event("ReqState")
                ori_func(this, seq_group, *args, **kwargs)
            return _allocate_and_set_running
        
        self.do_hook([Scheduler._allocate_and_set_running], allocate_and_set_running_maker)

        def preempt_by_recompute_maker(ori_func):
            def _preempt_by_recompute(this, seq_group, *args, **kwargs):
                seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
                if len(seqs) != 1:
                    Profiler(Level.INFO).res(seq_group.request_id).metric_inc('RUNNING', -1).\
                        metric_inc('WAITING', 1).event("ReqState")
                ori_func(this, seq_group, *args, **kwargs)
            return _preempt_by_recompute
        
        self.do_hook([Scheduler._preempt_by_recompute], preempt_by_recompute_maker)

        def swap_in_maker(ori_func):
            def _swap_in(this, seq_group, *args, **kwargs):
                Profiler(Level.INFO).res(seq_group.request_id).metric_inc('RUNNING', 1).\
                    metric_inc('SWAPPED', -1).event("ReqState")
                ori_func(this, seq_group, *args, **kwargs)
            return _swap_in
        
        self.do_hook([Scheduler._swap_in], swap_in_maker)

        def swap_out_maker(ori_func):
            def _swap_out(this, seq_group, *args, **kwargs):
                if self.block_manager.can_swap_out(seq_group):
                    Profiler(Level.INFO).res(seq_group.request_id).metric_inc('RUNNING', -1).\
                        metric_inc('SWAPPED', 1).event("ReqState")
                ori_func(this, seq_group, *args, **kwargs)
            return _swap_out
        
        self.do_hook([Scheduler._swap_out], swap_out_maker)

        def free_finished_seq_groups_maker(ori_func):
            def free_finished_seq_groups(this, *args, **kwargs):
                for seq_group in this.running:
                    Profiler(Level.INFO).res(seq_group.request_id).metric_inc('RUNNING', -1).\
                        metric_inc('FINISHED', 1).event("ReqState")
                    Profiler(Level.INFO).res(seq_group.request_id).event("Dequeue")
                ori_func(this, *args, **kwargs)
            return free_finished_seq_groups
        
        self.do_hook([Scheduler.free_finished_seq_groups], free_finished_seq_groups_maker)


class LLMEngineHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine

        def add_processed_request_maker(ori_func):
            def _add_processed_request(this, request_id, *args, **kwargs):
                ori_func(this, request_id, *args, **kwargs)
                Profiler(Level.INFO).res(request_id).event("Enqueue")
                Profiler(Level.INFO).res(request_id).metric_inc('WAITING', 1).event("ReqState")

            return _add_processed_request
        
        self.do_hook([LLMEngine._add_processed_request], add_processed_request_maker)


batch_hookers = [SchedulerHook, LLMEngineHook]