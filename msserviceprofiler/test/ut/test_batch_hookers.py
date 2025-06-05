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
import unittest
import sys
from unittest.mock import MagicMock, patch, call
from collections import deque


# 模拟 SequenceStatus 类
class FakeSequenceStatus:
    RUNNING = "RUNNING"


# 模拟 SequenceGroupMetadata 类
class FakeSequenceGroupMetadata:
    def __init__(self, request_id, token_chunk_size):
        self.request_id = request_id
        self.token_chunk_size = token_chunk_size
        self.fake_seqs = [FakeSequenceStatus(), FakeSequenceStatus()]

    def get_seqs(self, status):
        return self.fake_seqs


# 模拟 BlockManager 行为
class FakeBlockManager:
    @staticmethod
    def can_swap_out(self, seq_group=None):
        return True


# 模拟 Scheduler 类
class FakeScheduler:
    def __init__(self, sequence_group):
        self.sequence_group_list = [sequence_group, sequence_group]
        self.block_manager = FakeBlockManager()
        self.running = deque()
        self.waiting = deque()
        self.swapped = deque()

    def schedule(self):
        seq_group_metadata_list = self.sequence_group_list
        scheduler_outputs = self.sequence_group_list
        allow_async_output_proc = self.sequence_group_list
        return seq_group_metadata_list, scheduler_outputs, allow_async_output_proc

    def add_seq_group(self, seq_group):
        pass

    def abort_seq_group(self, request_id):
        pass

    def free_finished_seq_groups(self):
        pass

    def _allocate_and_set_running(self, seq_group):
        pass

    def _preempt_by_recompute(self, seq_group):
        pass

    def _swap_in(self, seq_group):
        pass

    def _swap_out(self, seq_group):
        pass

    def _add_seq_group_to_running(self, seq_group):
        pass

    def _schedule_priority_preemption(self, budget):
        pass

    def _schedule_default(self):
        pass

    def _schedule_chunked_prefill(self):
        pass

    def _add_seq_group_to_swapped(self, seq_group):
        pass


# 模拟 Scheduler 类
class FakeLLMEngine:
    def _add_processed_request(self, request_id):
        pass



# 导入被测试的类
from ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.batch_hookers import (
    Profiler, queue_profiler, Level
)


# 测试hook类
@patch('ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.batch_hookers.Profiler')
class TestSchedulerHook(unittest.TestCase):

    def setUp(self):
        # 将模拟的类和模块注入 sys.modules
        sys.modules['vllm.core.scheduler'] = MagicMock(Scheduler=FakeScheduler)
        sys.modules['vllm.engine.llm_engine'] = MagicMock(LLMEngine=FakeLLMEngine)
        sys.modules['vllm.sequence'] = MagicMock(
            SequenceGroupMetadata=FakeSequenceGroupMetadata,
            SequenceStatus=FakeSequenceStatus
        )

        # 导入被测试的类
        from ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.batch_hookers import (
            SchedulerHook, LLMEngineHook
        )

        # 初始化 Hook 实例
        self.scheduler_hook = SchedulerHook()
        self.llm_engine_hook = LLMEngineHook()

        # 调用 init 方法，其中的do_hook会将被测试的_maker函数与Fake类中的函数关联起来
        self.scheduler_hook.init()
        self.llm_engine_hook.init()

        # 定义测试参数变量
        self.fake_request_id = 0
        self.fake_seq_group = FakeSequenceGroupMetadata(self.fake_request_id, 0)
        self.fake_budget = 0

        # 初始化测试的Fake实例
        self.fake_scheduler = FakeScheduler(self.fake_seq_group)
        self.fake_llm_engine = FakeLLMEngine()

    # setUp中的init函数调用之后，调用Fake中原函数，会调用到对应的_maker函数
    # 校验_maker函数被正确调用，且其中的Profiler打点正确
    def test_schedule_maker(self, mock_profiler):
        seq_group_metadata_list, scheduler_outputs, allow_async_output_proc = \
            self.fake_scheduler.schedule()

        # 验证返回值是否正确
        self.assertEqual(seq_group_metadata_list, self.fake_scheduler.sequence_group_list)
        self.assertEqual(scheduler_outputs, self.fake_scheduler.sequence_group_list)
        self.assertEqual(allow_async_output_proc, self.fake_scheduler.sequence_group_list)

        # 验证Profiler方法正确调用且参数正确
        mock_profiler(Level.INFO).domain("BatchSchedule").span_start.assert_called_with("batchFrameworkProcessing")

    def test_abort_seq_group_maker(self, mock_profiler):
        self.fake_scheduler.abort_seq_group(self.fake_request_id)

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res(self.fake_request_id)
        expected_call = expected_call.metric_inc('FINISHED_ABORTED', 1).event("ReqState")
        mock_profiler.assert_has_calls([expected_call])

    def test_allocate_and_set_running_maker(self, mock_profiler):
        self.fake_scheduler._allocate_and_set_running(self.fake_seq_group)

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res(self.fake_request_id)
        expected_call = expected_call.metric_inc('RUNNING', 1).metric_inc('WAITING', -1).event("ReqState")
        mock_profiler.assert_has_calls([expected_call])

    def test_preempt_by_recompute_maker(self, mock_profiler):
        self.fake_scheduler._preempt_by_recompute(self.fake_seq_group)

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res(self.fake_request_id)
        expected_call = expected_call.metric_inc('RUNNING', -1).metric_inc('WAITING', 1).event("ReqState")
        mock_profiler.assert_has_calls([expected_call])

    def test_swap_in_maker(self, mock_profiler):
        self.fake_scheduler._swap_in(self.fake_seq_group)

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res(self.fake_request_id)
        expected_call = expected_call.metric_inc('RUNNING', 1).metric_inc('SWAPPED', -1).event("ReqState")
        mock_profiler.assert_has_calls([expected_call])

    def test_swap_out_maker(self, mock_profiler):
        self.fake_scheduler._swap_out(self.fake_seq_group)

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res(self.fake_request_id)
        expected_call = expected_call.metric_inc('RUNNING', -1).metric_inc('SWAPPED', 1).event("ReqState")
        mock_profiler.assert_has_calls([expected_call])

    def test_free_finished_seq_groups_maker(self, mock_profiler):
        # 配置 running 队列
        self.fake_scheduler.running.append(self.fake_seq_group)

        self.fake_scheduler.free_finished_seq_groups()

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res(self.fake_request_id)
        expected_call = expected_call.metric_inc('RUNNING', -1).metric_inc('FINISHED', 1).event("ReqState")
        mock_profiler.assert_has_calls([expected_call])

    # 校验Profiler.res中的request list正确
    def test_add_seq_group_to_running_maker(self, mock_profiler):
        self.fake_scheduler._add_seq_group_to_running(self.fake_seq_group)

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res([self.fake_request_id])
        expected_call = expected_call.metric("QueueSize", len(self.fake_scheduler.running)).metric_scope('running')
        expected_call = expected_call.event("Enqueue")
        mock_profiler.assert_has_calls([expected_call])

    def test_add_seq_group_maker(self, mock_profiler):
        self.fake_scheduler.add_seq_group(self.fake_seq_group)

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res([self.fake_request_id])
        expected_call = expected_call.metric("QueueSize", len(self.fake_scheduler.waiting)).metric_scope('waiting')
        expected_call = expected_call.event("Enqueue")
        mock_profiler.assert_has_calls([expected_call])

    def test_add_seq_group_to_swapped_maker(self, mock_profiler):
        self.fake_scheduler._add_seq_group_to_swapped(self.fake_seq_group)

        # 校验Profiler方法链式调用正确
        expected_call = call(Level.INFO).domain("BatchSchedule").res([self.fake_request_id])
        expected_call = expected_call.metric("QueueSize", len(self.fake_scheduler.swapped)).metric_scope('swapped')
        expected_call = expected_call.event("Enqueue")
        mock_profiler.assert_has_calls([expected_call])
        mock_profiler(Level.INFO).domain("BatchSchedule").res.assert_called_with([self.fake_request_id])

    # 校验队列打点函数queue_profiler正确调用
    @patch('ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.batch_hookers.queue_profiler')
    def test_schedule_priority_preemption_maker(self, mock_queue_profiler, mock_profiler):
        self.fake_scheduler._schedule_priority_preemption(self.fake_budget)
        mock_queue_profiler.assert_called()

    @patch('ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.batch_hookers.queue_profiler')
    def test_schedule_default_maker(self, mock_queue_profiler, mock_profiler):
        self.fake_scheduler._schedule_default()
        mock_queue_profiler.assert_called()

    @patch('ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.batch_hookers.queue_profiler')
    def test_schedule_chunked_prefill_maker(self, mock_queue_profiler, mock_profiler):
        self.fake_scheduler._schedule_chunked_prefill()
        mock_queue_profiler.assert_called()

    # 测试LLMEngineHook中_maker函数
    def test_add_processed_request_maker(self, mock_profiler):
        self.fake_llm_engine._add_processed_request(self.fake_request_id)
        mock_profiler(Level.INFO).domain("Request").res.assert_called_with(self.fake_request_id)

    def test_queue_profiler(self, mock_profiler):
        # 测试seq1出队时的打点
        seq1 = FakeSequenceGroupMetadata(0, 1)
        seq2 = FakeSequenceGroupMetadata(1, 1)
        before_queue = deque([seq1, seq2])
        after_queue = deque([seq2])
        queue_profiler(before_queue, after_queue, "test_queue")

        # 验证 Profiler 正确调用且参数正确
        mock_profiler(Level.INFO).domain("BatchSchedule").res.assert_called_once()
        mock_profiler(Level.INFO).domain("BatchSchedule").res.call_args[0][0] == 1
