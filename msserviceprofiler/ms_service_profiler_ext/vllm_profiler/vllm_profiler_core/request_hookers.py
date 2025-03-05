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
import time

from vllm.sequence import SequenceGroupMetadata
from ms_service_profiler import Profiler, Level
from .vllm_hooker_base import VLLMHookerBase


# generate -> add_request -> schedule -> execute_model
# 在请求进入引擎时记录时间戳，用于后续计算队列等待时间。
class EngineRequestTrackerHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        def add_request_maker(ori_func):
            def add_request(this, request_id, prompt, *args, **kwargs):
                # 记录请求进入系统的时间                
                profiler = Profiler(Level.INFO)
                profiler.domain("http").res(request_id).metric(
                    "timestamp", time.time()).event("httpReq")              
                return ori_func(this, request_id, prompt, *args, **kwargs)
            return add_request

        self.do_hook([LLMEngine.add_request, AsyncLLMEngine.add_request], add_request_maker)


# 通过调度器获取 prefill/decode 阶段的请求元数据
class SchedulerHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine
        from vllm.core.scheduler import Scheduler

        def schedule_maker(ori_func):
            def schedule(this, *args, **kwargs):
                # 调用原始调度方法
                result = ori_func(this, *args, **kwargs)

                # 提取 prefill/decode 请求
                for seq_group in result[0]:
                    request_id = seq_group.request_id
                    input_token_ids = seq_group.seq_data
                    input_tokens = len(input_token_ids)

                    # 判断阶段 (prefill or decode)
                    if seq_group.is_prompt: 
                        profiler = Profiler(Level.INFO)
                        profiler.domain("http").res(request_id).metric(
                            "timestamp", time.time()).event("PrefillStart")
                    else:
                        profiler = Profiler(Level.INFO)
                        profiler.domain("http").res(request_id).metric(
                            "timestamp", time.time()).event("DecodeStart")
                return result
            return schedule

        # Hook 调度器的 schedule 方法
        self.do_hook([Scheduler.schedule], schedule_maker)


# 捕获请求完成或失败事件
class ServerGenerateHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.utils import iterate_with_cancellation

        cache_gen_2_req_id = {}  # Recording request_id in `generate`, and save in `iterate_with_cancellation_maker`

        def engine_generate_maker(ori_func):
            def generate(this, prompt, sampling_params, request_id, *args, **kwargs):
                try:
                    # 记录请求开始处理时间
                    profiler = Profiler(Level.INFO)
                    profiler.domain("http").res(request_id).metric(
                        "timestamp", time.time()).event("ProcessingStart")
                    ret = ori_func(this, prompt, sampling_params, request_id, *args, **kwargs)
                    cache_gen_2_req_id[id(ret)] = request_id
                    return ret
                except Exception as e:
                    profiler = Profiler(Level.INFO)
                    profiler.domain("http").res(request_id).metric(
                        "timestamp", time.time()).metric(
                        "error_type", type(e).__name__).metric(
                        "error_message", str(e)).event("RequestFailed")
                    raise

            return generate

        self.do_hook([AsyncLLMEngine.generate], engine_generate_maker)

        def iterate_with_cancellation_maker(ori_func):
            async def iterate_with_cancellation(iterator, is_cancelled, *args, **kwargs):
                request_id = cache_gen_2_req_id.get(id(iterator), None)
                try:
                    async for out in ori_func(iterator, is_cancelled, *args, **kwargs):
                        yield out
                except Exception as e:
                    if request_id is not None:
                        profiler = Profiler(Level.INFO)
                        profiler.domain("http").res(request_id).metric(
                            "timestamp", time.time()).metric(
                            "error_type", type(e).__name__).metric(
                            "error_message", str(e)).event("RequestFailed")
                    raise
                finally:
                    if request_id is not None:
                        # 记录完成事件
                        profiler = Profiler(Level.INFO)
                        profiler.domain("http").res(request_id).metric(
                            "timestamp", time.time()).event("httpRes")
                        cache_gen_2_req_id.pop(id(iterator), None)

            return iterate_with_cancellation

        self.do_hook([iterate_with_cancellation], iterate_with_cancellation_maker, pname="generate")


class LLMEngineHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine

        def get_stats_maker(ori_func):
            def get_stats(this, *args, **kwargs):
                profiler = Profiler(Level.INFO)
                stats = ori_func(this, *args, **kwargs)
                profiler.domain("http").metric(
                    "recvTokenSize", stats.num_prompt_tokens_iter).metric(
                    "replyTokenSize", stats.num_generation_tokens_iter).event("GetTokenSize")
                return stats
            return get_stats

        self.do_hook([LLMEngine._get_stats], get_stats_maker)


request_hookers = [EngineRequestTrackerHook, SchedulerHook, ServerGenerateHook, LLMEngineHook]
