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
import sys
import unittest
from unittest.mock import MagicMock, patch, call


# 模拟 SequenceGroupMetadata 类
class FakeSequenceGroupMetadata:
    def __init__(self, request_id):
        self.request_id = request_id


# 模拟 LLMEngine 类
class FakeLLMEngine:
    def __init__(self):
        self.stats = MagicMock(num_prompt_tokens_iter=10, num_generation_tokens_iter=5)

    def add_request(self, request_id, prompt):
        pass

    def _get_stats(self):
        return self.stats


# 模拟 AsyncLLMEngine 类
class FakeAsyncLLMEngine:
    def add_request(self, request_id, prompt):
        pass


# 模拟 iterate_with_cancellation 函数
async def fake_iterate_with_cancellation(iterator, is_cancelled):
    yield "output"


# 将模拟的类和模块注入 sys.modules
sys.modules['vllm.engine.llm_engine'] = MagicMock(LLMEngine=FakeLLMEngine)
sys.modules['vllm.engine.async_llm_engine'] = MagicMock(AsyncLLMEngine=FakeAsyncLLMEngine)
sys.modules['vllm.sequence'] = MagicMock(SequenceGroupMetadata=FakeSequenceGroupMetadata)

# 导入被测试的类
from ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.request_hookers import (
    EngineRequestTrackerHook, LLMEngineHook,
    Profiler, Level
)


@patch('ms_service_profiler_ext.vllm_profiler.vllm_profiler_core.request_hookers.Profiler')
class TestVLLMHookers(unittest.TestCase):

    def setUp(self):
        # 初始化测试的Fake实例
        self.fake_llm_engine = FakeLLMEngine()
        self.fake_async_llm_engine = FakeAsyncLLMEngine()
        self.fake_request_id = 123
        self.fake_prompt = "test_prompt"

    def test_engine_request_tracker_hook(self, mock_profiler):
        # 初始化 EngineRequestTrackerHook
        engine_request_tracker_hook = EngineRequestTrackerHook()
        engine_request_tracker_hook.init()

        # 调用 add_request 方法
        self.fake_llm_engine.add_request(self.fake_request_id, self.fake_prompt)

        # 验证 Profiler 调用
        expected_call = call(Level.INFO).domain("http").res(self.fake_request_id).metric(
            "timestamp", unittest.mock.ANY).event("httpReq")
        mock_profiler.assert_has_calls([expected_call])

    def test_llm_engine_hook(self, mock_profiler):
        # 初始化 LLMEngineHook
        llm_engine_hook = LLMEngineHook()
        llm_engine_hook.init()

        # 调用 _get_stats 方法
        stats = self.fake_llm_engine._get_stats()

        # 验证 Profiler 调用
        expected_call = call(Level.INFO).domain("http").metric(
            "recvTokenSize", 10).metric(
            "replyTokenSize", 5).event("GetTokenSize")
        mock_profiler.assert_has_calls([expected_call])
