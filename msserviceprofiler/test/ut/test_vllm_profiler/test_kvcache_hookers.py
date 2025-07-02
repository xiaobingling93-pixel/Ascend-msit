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
import os
import sys
import unittest
import pkgutil
from unittest.mock import MagicMock, patch, call
from collections import namedtuple

# skip importing from __init__
sys.path.append(os.path.join(os.path.dirname(pkgutil.get_loader("msserviceprofiler").path), "vllm_profiler"))
from vllm_profiler_core.kvcache_hookers import Profiler, Level


# 模拟 SequenceGroupMetadata 类
class FakeSequenceGroupMetadata:
    def __init__(self, request_id, seqs):
        self.request_id = request_id
        self.seqs = seqs


# 模拟 Sequence 类
class FakeSequence:
    def __init__(self, seq_id):
        self.seq_id = seq_id


# 模拟 SelfAttnBlockSpaceManager 类
class FakeSelfAttnBlockSpaceManager:
    def __init__(self):
        self.block_tables = {0: [1, 2, 3]}  # 模拟 block_tables

    @staticmethod
    def get_num_free_gpu_blocks():
        return 1

    def allocate(self, seq_group):
        self.block_tables = {0: [1]}
        pass

    def append_slots(self, seq, num_lookahead_slots):
        self.block_tables = {0: [2]}
        return 1  # 模拟 new_cows

    def swap_in(self, seq_group):
        self.block_tables = {0: [1, 2]}
        return True  # 模拟 swap_in 结果

    def swap_out(self, seq_group):
        self.block_tables = {0: [1]}
        return True  # 模拟 swap_out 结果

    def free(self, seq):
        self.block_tables = {}
        pass


# 模拟 Stats 类
class FakeStats:
    def __init__(self):
        self.cpu_cache_usage_sys = 0.5
        self.gpu_cache_usage_sys = 0.8


# 模拟 LLMEngine 类
class FakeLLMEngine:
    def __init__(self):
        self.stats = FakeStats()
        self.scheduler = [namedtuple("scheduler", ["block_manager"])(FakeSelfAttnBlockSpaceManager())]

    def _get_stats(self):
        return self.stats


@patch("vllm_profiler_core.kvcache_hookers.Profiler")
class TestKVCacheManagerHook(unittest.TestCase):

    def setUp(self):
        # 将模拟的类和模块注入 sys.modules
        sys.modules["vllm.core.block_manager"] = MagicMock(SelfAttnBlockSpaceManager=FakeSelfAttnBlockSpaceManager)
        sys.modules["vllm.engine.llm_engine"] = MagicMock(LLMEngine=FakeLLMEngine)

        # 导入被测试的类
        from vllm_profiler_core.kvcache_hookers import KVCacheManagerHook

        # 初始化 Hook 实例
        self.kvcache_hook = KVCacheManagerHook()

        # 调用 init 方法，定义hook函数
        self.kvcache_hook.init()

        # 定义测试参数变量
        self.fake_request_id = 0
        self.fake_seq_id = 1
        self.fake_seq = FakeSequence(self.fake_seq_id)
        self.fake_seqs = [self.fake_seq]
        self.fake_seq_group = FakeSequenceGroupMetadata(self.fake_request_id, self.fake_seqs)

        # 初始化测试的Fake实例
        self.fake_block_manager = FakeSelfAttnBlockSpaceManager()
        self.fake_llm_engine = FakeLLMEngine()

    def test_allocate_maker(self, mock_profiler):
        self.fake_block_manager.allocate(self.fake_seq_group)
        expected_call = (
            call(Level.INFO)
            .domain("KVCache")
            .res(0)
            .metric("deviceBlock", len(self.fake_block_manager.block_tables))
            .event("Allocate")
        )
        mock_profiler.assert_has_calls([expected_call])

    def test_append_slots_maker(self, mock_profiler):
        new_cows = self.fake_block_manager.append_slots(self.fake_seq, 1)
        self.assertEqual(new_cows, 1)
        expected_call = (
            call(Level.INFO)
            .domain("KVCache")
            .res(0)
            .metric("deviceBlock", len(self.fake_block_manager.block_tables))
            .event("AppendSlot")
        )
        mock_profiler.assert_has_calls([expected_call])

    def test_swap_in_maker(self, mock_profiler):
        res = self.fake_block_manager.swap_in(self.fake_seq_group)
        self.assertTrue(res)
        expected_call = (
            call(Level.INFO)
            .domain("KVCache")
            .res(0)
            .attr("swap", "swap_in")
            .metric("deviceBlock", len(self.fake_block_manager.block_tables))
            .event("SwapIn")
        )
        mock_profiler.assert_has_calls([expected_call])

    def test_swap_out_maker(self, mock_profiler):
        res = self.fake_block_manager.swap_out(self.fake_seq_group)
        self.assertTrue(res)
        expected_call = (
            call(Level.INFO)
            .domain("KVCache")
            .res(0)
            .attr("swap", "swap_out")
            .metric("deviceBlock", len(self.fake_block_manager.block_tables))
            .event("SwapOut")
        )
        mock_profiler.assert_has_calls([expected_call])

    def test_free_maker(self, mock_profiler):
        self.fake_block_manager.free(self.fake_seq)
        expected_call = (
            call(Level.INFO)
            .domain("KVCache")
            .res(0)
            .metric("deviceBlock", len(self.fake_block_manager.block_tables))
            .event("Free")
        )
        mock_profiler.assert_has_calls([expected_call])

    def test_get_stats_maker(self, mock_profiler):
        stats = self.fake_llm_engine._get_stats()
        self.assertEqual(stats.cpu_cache_usage_sys, 0.5)
        self.assertEqual(stats.gpu_cache_usage_sys, 0.8)
