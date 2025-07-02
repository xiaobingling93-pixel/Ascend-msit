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
from dataclasses import dataclass
from collections import namedtuple
from unittest.mock import MagicMock, patch, call
# skip importing from __init__
sys.path.append(os.path.join(os.path.dirname(pkgutil.get_loader("msserviceprofiler").path), "vllm_profiler"))
from vllm_profiler_core.request_hookers import Profiler, Level


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


@patch("vllm_profiler_core.request_hookers.Profiler")
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
        from vllm_profiler_core.request_hookers import EngineRequestTrackerHook063

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
        from vllm_profiler_core.request_hookers import EngineRequestTrackerHook084

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
        from vllm_profiler_core.request_hookers import LLMEngineHook063

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

    def test_llm_engine_hook_084(self, mock_profiler):
        # 导入被测试的类
        from vllm_profiler_core.request_hookers import LLMEngineHook084

        # 模拟必要的类和结构
        FakeScheduledSeqGroup = namedtuple('ScheduledSeqGroup', ['seq_group'])
        FakeSeqGroup = namedtuple('SeqGroup', ['is_finished', 'seqs'])
        FakeSequence = namedtuple('Sequence', ['get_prompt_len', 'get_output_len'])
        
        # 创建测试数据
        fake_seq = FakeSequence(get_prompt_len=lambda: 5, get_output_len=lambda: 3)
        fake_seq_group = FakeSeqGroup(is_finished=lambda: True, seqs=[fake_seq])
        fake_scheduled_seq_group = FakeScheduledSeqGroup(seq_group=fake_seq_group)
        
        fake_metadata = MagicMock(request_id="123")
        fake_scheduler_outputs = MagicMock(scheduled_seq_groups=[fake_scheduled_seq_group])
        
        # 模拟 Context 对象
        fake_ctx = MagicMock()
        fake_ctx.output_queue = [(None, [fake_metadata], fake_scheduler_outputs, None, None, None, [])]
        
        # 初始化 LLMEngineHook084
        llm_engine_hook = LLMEngineHook084()
        llm_engine_hook.init()

        # 调用 _process_model_outputs 方法
        self.fake_llm_engine._process_model_outputs(fake_ctx, request_id="123")

        # 验证 Profiler 调用
        expected_calls = [
            call(Level.INFO).domain("Request").res("123"),
            call(Level.INFO).domain("Request").res("123"),
            call(Level.INFO).domain("Request").res("123"),
            call(Level.INFO).domain("Request").res(["123"]).event("DecodeEnd"),
            call(Level.INFO).domain("Request").res("123").metric("recvTokenSize", 5).event("httpRes"),
            call(Level.INFO).domain("Request").res("123").metric("replyTokenSize", 3).event("httpRes")
        ]
        
        # 检查所有预期的调用
        for expected_call in expected_calls:
            mock_profiler.assert_has_calls([expected_call], any_order=True)