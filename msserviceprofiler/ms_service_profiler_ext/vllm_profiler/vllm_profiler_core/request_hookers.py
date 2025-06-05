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
import time

from ms_service_profiler import Profiler, Level
from .vllm_hooker_base import VLLMHookerBase

GLOBAL_FORWARD_PROF = []


def prof_add_request(request_id, prompt, *args, **kwargs):
    # 记录请求进入系统的时间
    Profiler(Level.INFO).domain("Request").res(request_id).event("httpReq")
    Profiler(Level.INFO).domain("Request").res(request_id).event("encode")


# generate -> add_request -> schedule -> execute_model
# 在请求进入引擎时记录时间戳，用于后续计算队列等待时间。
class EngineRequestTrackerHook063(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        def add_request_maker(ori_func):
            def add_request(this, request_id, prompt, *args, **kwargs):
                prof_add_request(request_id, prompt, *args, **kwargs)
                return ori_func(this, request_id, prompt, *args, **kwargs)

            return add_request

        self.do_hook([LLMEngine.add_request, AsyncLLMEngine.add_request], add_request_maker)


class EngineRequestTrackerHook084(VLLMHookerBase):
    vllm_version = ("0.8.4", "0.8.4")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        def add_request_maker(ori_func):
            def add_request(this, request_id, prompt, *args, **kwargs):
                prof_add_request(request_id, prompt, *args, **kwargs)
                return ori_func(this, request_id, prompt, *args, **kwargs)

            return add_request

        def add_request_async_maker(ori_func):
            async def add_request(this, request_id, prompt, *args, **kwargs):
                prof_add_request(request_id, prompt, *args, **kwargs)
                return ori_func(this, request_id, prompt, *args, **kwargs)

            return add_request

        self.do_hook([LLMEngine.add_request], add_request_maker)
        self.do_hook([AsyncLLMEngine.add_request], add_request_async_maker)


# 采集请求返回的数据
class LLMEngineHook084(VLLMHookerBase):
    vllm_version = ("0.8.4", "0.8.4")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine

        def process_model_outputs_maker(ori_func):
            def process_model_outputs(this, ctx, request_id=None, *args, **kwargs):
                if len(ctx.output_queue) == 0:
                    return ori_func(this, ctx, request_id, *args, **kwargs)

                request_id_list = []
                outputs, metadata_list, scheduler_outputs, _, _, _, skip = ctx.output_queue[0]
                for ii, (sch_seq_group, meta) in enumerate(zip(scheduler_outputs.scheduled_seq_groups, metadata_list)):
                    if ii in skip:
                        continue
                    seq_group, seq_request_id = sch_seq_group.seq_group, meta.request_id
                    request_id_list.append(seq_request_id)
                    if seq_group.is_finished() and len(seq_group.seqs) > 0:
                        cur_seq = seq_group.seqs[0]
                        profiler_recv = Profiler(Level.INFO).domain("Request").res(seq_request_id)
                        profiler_reply = Profiler(Level.INFO).domain("Request").res(seq_request_id)
                        profiler_recv.metric("recvTokenSize", cur_seq.get_prompt_len()).event("httpRes")
                        profiler_reply.metric("replyTokenSize", cur_seq.get_output_len()).event("httpRes")

                prof = Profiler(Level.INFO).domain("Request").res(request_id_list)
                ret = ori_func(this, ctx, request_id, *args, **kwargs)
                prof.event("DecodeEnd")
                return ret

            return process_model_outputs

        self.do_hook([getattr(LLMEngine, "_process_model_outputs")], process_model_outputs_maker)


class LLMEngineHook063(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine

        def validate_output_maker(ori_func):
            def validate_output(output, output_type):
                profiler_recv = Profiler(Level.INFO).domain("Request")
                profiler_reply = Profiler(Level.INFO).domain("Request")
                if output.finished is True:
                    request_id = output.request_id
                    input_token_size = len(output.prompt_token_ids)
                    output_token_size = len(output.outputs[0].token_ids)
                    profiler_recv.res(request_id).metric("recvTokenSize", input_token_size).event("httpRes")
                    profiler_reply.res(request_id).metric("replyTokenSize", output_token_size).event("httpRes")
                return ori_func(output, output_type)

            return validate_output

        self.do_hook([LLMEngine.validate_output], validate_output_maker)


request_hookers = [EngineRequestTrackerHook063, EngineRequestTrackerHook084, LLMEngineHook063, LLMEngineHook084]
