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
from dataclasses import dataclass
from collections import namedtuple
from unittest.mock import MagicMock, patch, call
from msserviceprofiler.vllm_profiler.vllm_profiler_core.request_hookers import Profiler, Level


# 模拟 RequestOutput 类
class FakeRequestOutput:
    def __init__(self):
        self.request_id = "123"
        self.prompt_token_ids = [0, 1, 2, 3, 4]
        self.finished = True
        self.outputs = [FakeCompletionOutput]


@dataclass
class FakeCompletionOutput:
    token_ids = (0, 1, 2)


# 模拟 LLMEngine 类
class FakeLLMEngine:
    def __init__(self):
        pass

    @classmethod
    def validate_output(cls, output, output_type):
        pass

    def add_request(self, request_id, prompt):
        pass

    def _process_model_outputs(self, ctx=None, request_id=None):
        pass


# 模拟 AsyncLLMEngine 类
class FakeAsyncLLMEngine:
    def add_request(self, request_id, prompt):
        pass


@patch("msserviceprofiler.vllm_profiler.vllm_profiler_core.request_hookers.Profiler")
class TestVLLMHookers(unittest.TestCase):

    def setUp(self):
        # 将模拟的类和模块注入 sys.modules
        sys.modules["vllm.engine.llm_engine"] = MagicMock(LLMEngine=FakeLLMEngine)
        sys.modules["vllm.engine.async_llm_engine"] = MagicMock(AsyncLLMEngine=FakeAsyncLLMEngine)

        # 初始化测试的Fake实例
        self.fake_llm_engine = FakeLLMEngine()
        self.fake_async_llm_engine = FakeAsyncLLMEngine()
        self.fake_request_id = 123
        self.fake_prompt = "test_prompt"
        self.fake_output = FakeRequestOutput()
        self.fake_output_type = object

    def test_engine_request_tracker_hook_063(self, mock_profiler):
        # 导入被测试的类
        from msserviceprofiler.vllm_profiler.vllm_profiler_core.request_hookers import EngineRequestTrackerHook063

        # 初始化 EngineRequestTrackerHook
        engine_request_tracker_hook = EngineRequestTrackerHook063()
        engine_request_tracker_hook.init()

        # 调用 add_request 方法
        self.fake_llm_engine.add_request(self.fake_request_id, self.fake_prompt)

        # 验证 Profiler 调用
        expected_call = call(Level.INFO).domain("Request").res(self.fake_request_id).event("httpReq")
        mock_profiler.assert_has_calls([expected_call])

    def test_engine_request_tracker_hook_084(self, mock_profiler):
        # 导入被测试的类
        from msserviceprofiler.vllm_profiler.vllm_profiler_core.request_hookers import EngineRequestTrackerHook084

        # 初始化 EngineRequestTrackerHook
        engine_request_tracker_hook = EngineRequestTrackerHook084()
        engine_request_tracker_hook.init()

        # 调用 add_request 方法
        self.fake_llm_engine.add_request(self.fake_request_id, self.fake_prompt)

        # 验证 Profiler 调用
        expected_call = call(Level.INFO).domain("Request").res(self.fake_request_id).event("httpReq")
        mock_profiler.assert_has_calls([expected_call])

    def test_llm_engine_hook_063(self, mock_profiler):
        # 导入被测试的类
        from msserviceprofiler.vllm_profiler.vllm_profiler_core.request_hookers import LLMEngineHook063

        # 初始化 LLMEngineHook
        llm_engine_hook = LLMEngineHook063()
        llm_engine_hook.init()

        # 调用 validate_output 方法
        self.fake_llm_engine.validate_output(self.fake_output, self.fake_output_type)

        # 验证 Profiler 调用
        expected_call_1 = call(Level.INFO).domain("Request").res("123").metric("recvTokenSize", 5).event("httpRes")
        expected_call_2 = call(Level.INFO).domain("Request").res("123").metric("replyTokenSize", 3).event("httpRes")
        mock_profiler.assert_has_calls([expected_call_1])
        mock_profiler.assert_has_calls([expected_call_2])
