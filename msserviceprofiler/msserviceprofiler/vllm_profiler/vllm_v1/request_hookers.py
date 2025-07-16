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
from ..module_hook import vllm_hook


@vllm_hook(("vllm.engine.async_llm_engine", "AsyncLLMEngine.add_request"), min_version="0.9.1")
async def add_request_async(original_func, this, request_id, prompt, *args, **kwargs):
    Profiler(Level.INFO).domain("Request").res(request_id).event("httpReq")
    Profiler(Level.INFO).domain("Request").res(request_id).event("encode")
    return await original_func(this, request_id, prompt, *args, **kwargs)

@vllm_hook(("vllm.engine.llm_engine", "LLMEngine._process_model_outputs"), min_version="0.8.4")
def process_model_outputs(original_func, this, ctx, request_id=None, *args, **kwargs):
    if len(ctx.output_queue) == 0:
        return original_func(this, ctx, request_id, *args, **kwargs)
    outputs, metadata_list, scheduler_outputs, _, _, _, skip = ctx.output_queue[0]
    ret = original_func(this, ctx, request_id, *args, **kwargs)

    request_id_list = []
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
    prof.event("DecodeEnd")
    return ret
@vllm_hook(("vllm.v1.engine.output_processor", "OutputProcessor.process_outputs"), min_version="0.9.1")
def process_outputs(original_func, this, engine_core_outputs, *args, **kwargs):
    if len(engine_core_outputs) == 0:
        return original_func(this, engine_core_outputs, *args, **kwargs)

    request_id_list = []
    for engine_core_output in engine_core_outputs:
        request_id = engine_core_output.request_id
        request_state = this.request_states.get(request_id)
        request_id_list.append(request_id)
        if request_state and engine_core_output.finish_reason is not None:
            recv_token_size = len(request_state.prompt_token_ids)
            reply_token_size = (request_state.stats.num_generation_tokens if request_state.stats else None)
            
            profiler_recv = Profiler(Level.INFO).domain("Request").res(request_id)
            profiler_reply = Profiler(Level.INFO).domain("Request").res(request_id)
            profiler_recv.metric("recvTokenSize", recv_token_size).event("httpRes")
            profiler_reply.metric("replyTokenSize", reply_token_size).event("httpRes")

    ret = original_func(this, engine_core_outputs, *args, **kwargs)
    prof = Profiler(Level.INFO).domain("Request").res(request_id_list)
    prof.event("DecodeEnd")
    return ret
