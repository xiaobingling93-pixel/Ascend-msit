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
from .vllm_hooker_base import VLLMHookerBase
from .profiling_csv_writer import profiling_csv_writer as pp


class ModelRunnerExecuteHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.worker.model_runner import ModelRunner

        def execute_model_maker(ori_func):
            def execute_model(this, model_input, kv_caches, *args, **kwargs):
                start_time = time.time()
                is_prefill = not kv_caches
                ret = ori_func(this, model_input, kv_caches, *args, **kwargs)
                duration = time.time() - start_time

                http_rids = []
                recv_tokens_map = {}
                if getattr(model_input, "seq_group_metadata_list", None):
                    for seq_group_meta in model_input.seq_group_metadata_list:
                        if isinstance(seq_group_meta, SequenceGroupMetadata):
                            http_rid = seq_group_meta.request_id
                            http_rids.append(http_rid)
                            if is_prefill:
                                recv_tokens = len(seq_group_meta.input_token_ids)
                                recv_tokens_map[http_rid] = recv_tokens

                if is_prefill:
                    for http_rid in http_rids:
                        pp.put_event(
                            {"event_type": "start_process", "http_rid": http_rid, "timestamp": start_time, "data": {}}
                        )
                        pp.put_event(
                            {
                                "event_type": "prefill",
                                "http_rid": http_rid,
                                "timestamp": time.time(),
                                "data": {"duration": duration, "recv_tokens": recv_tokens_map.get(http_rid, 0)},
                            }
                        )
                else:
                    for http_rid in http_rids:
                        pp.put_event(
                            {
                                "event_type": "decode",
                                "http_rid": http_rid,
                                "timestamp": time.time(),
                                "data": {"duration": duration},
                            }
                        )
                return ret

            return execute_model

        self.do_hook([ModelRunner.execute_model], execute_model_maker)


class EngineAddRequestHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.llm_engine import LLMEngine
        from vllm.engine.async_llm_engine import RequestTracker

        def add_requests_maker(ori_func):
            def add_request(this, http_rid, *args, **kwargs):
                pp.put_event({"event_type": "start", "http_rid": http_rid, "timestamp": time.time(), "data": {}})
                return ori_func(this, http_rid, *args, **kwargs)

            return add_request

        self.do_hook([LLMEngine.add_request, RequestTracker.add_request], add_requests_maker)


class ServerGenerateHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.utils import iterate_with_cancellation

        cache_gen_2_req_id = {}

        def engine_generate_maker(ori_func):
            def generate(this, prompt, sampling_params, request_id, *args, **kwargs):
                try:
                    ret = ori_func(this, prompt, sampling_params, request_id, *args, **kwargs)
                    cache_gen_2_req_id[id(ret)] = request_id
                    return ret
                except Exception as e:
                    pp.put_event(
                        {
                            "event_type": "error",
                            "http_rid": request_id,
                            "timestamp": time.time(),
                            "data": {"error_type": type(e).__name__, "error_message": str(e)},
                        }
                    )
                    raise

            return generate

        self.do_hook([AsyncLLMEngine.generate], engine_generate_maker)

        def iterate_with_cancellation_maker(ori_func):
            async def iterate_with_cancellation(iterator, is_cancelled, *args, **kwargs):
                http_rid = cache_gen_2_req_id.get(id(iterator), None)
                try:
                    async for out in ori_func(iterator, is_cancelled, *args, **kwargs):
                        yield out
                except Exception as e:
                    if http_rid is not None:
                        pp.put_event(
                            {
                                "event_type": "error",
                                "http_rid": http_rid,
                                "timestamp": time.time(),
                                "data": {"error_type": type(e).__name__, "error_message": str(e)},
                            }
                        )
                    raise
                finally:
                    if http_rid is not None:
                        pp.put_event(
                            {"event_type": "finish", "http_rid": http_rid, "timestamp": time.time(), "data": {}}
                        )
                        cache_gen_2_req_id.pop(id(iterator), None)

            return iterate_with_cancellation

        self.do_hook([iterate_with_cancellation], iterate_with_cancellation_maker, pname="generate")


all_hookers = [ModelRunnerExecuteHook, EngineAddRequestHook, ServerGenerateHook]
