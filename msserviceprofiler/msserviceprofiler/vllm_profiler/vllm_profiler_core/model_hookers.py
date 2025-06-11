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
from .vllm_hooker_base import VLLMHookerBase

GLOBAL_FORWARD_PROF = []


class ExecutorBaseExecuteModelHook(VLLMHookerBase):
    vllm_version = ("0.8.4", "0.8.4")

    def init(self):
        from vllm.executor.executor_base import ExecutorBase
        from vllm.executor.executor_base import DistributedExecutorBase

        def execute_model_maker(ori_func):
            def execute_model(this, execute_model_req, *args, **kwargs):
                prof = Profiler(Level.INFO).domain("ModelExecute")
                is_prefill, request_id_list, request_id_with_iter_list = False, [], []
                for seq_metadata in execute_model_req.seq_group_metadata_list:
                    if len(seq_metadata.seq_data) > 0:
                        cur_seq_data = list(seq_metadata.seq_data.values())[0]
                        iter_size = cur_seq_data.get_len() - len(cur_seq_data.prompt_token_ids)
                    else:
                        iter_size = 0
                    request_id_list.append({"rid": seq_metadata.request_id})
                    request_id_with_iter_list.append({"rid": seq_metadata.request_id, "iter_size": iter_size})
                    is_prefill = is_prefill or seq_metadata.is_prompt

                prof.res(request_id_with_iter_list)
                prof.attr("batch_type", "Prefill" if is_prefill else "Decode")
                prof.span_start("modelExec")
                prof.attr("batch_size", len(execute_model_req.seq_group_metadata_list))

                preprocess_prof = Profiler(Level.INFO).domain("ModelExecute").res(request_id_list)
                preprocess_prof.event("preprocess")

                ret = ori_func(this, execute_model_req, *args, **kwargs)
                prof.span_end()
                return ret

            return execute_model

        self.do_hook([ExecutorBase.execute_model], execute_model_maker)
        self.do_hook([DistributedExecutorBase.execute_model], execute_model_maker)


class ModelRunnerExecuteHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.8.4")

    def __init__(self):
        super().__init__()
        self.is_model_first_run = True

    def init(self):
        from vllm.worker.model_runner import ModelRunner

        def execute_model_maker(ori_func):
            def execute_model(this, model_input, kv_caches, *args, **kwargs):
                if self.is_model_first_run:
                    self.is_model_first_run = False
                    return ori_func(this, model_input, kv_caches, *args, **kwargs)

                prof = Profiler(Level.INFO).domain("ModelExecute")
                prof.span_start("modelExec")

                ret = ori_func(this, model_input, kv_caches, *args, **kwargs)

                is_prefill = model_input.attn_metadata.prefill_metadata

                request_id_list = []

                for request_id, _ in model_input.request_ids_to_seq_ids.items():
                    request_id_list.append({"rid": request_id})

                prof.res(request_id_list)

                if is_prefill:
                    prof.attr("batch_type", "Prefill")
                else:
                    prof.attr("batch_type", "Decode")

                batch_size = model_input.input_tokens.shape[0]
                prof.attr("batch_size", batch_size)

                prof.span_end()
                return ret

            return execute_model

        self.do_hook([ModelRunner.execute_model], execute_model_maker)


class ModelForwardHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.8.4")

    def __init__(self):
        super().__init__()
        self.is_forward_first_run = True

    def init(self):
        from vllm.attention.backends.utils import CommonAttentionState

        def begin_forward_maker(ori_func):
            def begin_forward(this, model_input, *args, **kwargs):
                ret = ori_func(this, model_input, *args, **kwargs)
                if self.is_forward_first_run:
                    self.is_forward_first_run = False
                    return ret

                request_id_list = [{"rid": request_id} for request_id, _ in model_input.request_ids_to_seq_ids.items()]
                prof = Profiler(Level.INFO).domain("ModelExecute").res(request_id_list)
                GLOBAL_FORWARD_PROF.append(prof)
                return ret

            return begin_forward

        self.do_hook([CommonAttentionState.begin_forward], begin_forward_maker)


class SetForwardContextHook(VLLMHookerBase):
    vllm_version = ("0.8.4", "0.8.4")

    def init(self):
        from vllm import forward_context
        from contextlib import contextmanager

        def set_forward_context_maker(ori_func):
            @contextmanager
            def set_forward_context(*args, **kwargs):
                if len(GLOBAL_FORWARD_PROF) > 0:
                    prof = GLOBAL_FORWARD_PROF.pop(0)
                    prof.span_start("forward")
                else:
                    prof = None
                with ori_func(*args, **kwargs):
                    yield
                if prof is not None:
                    prof.span_end()

            return set_forward_context

        self.do_hook([forward_context.set_forward_context], set_forward_context_maker)


model_hookers = [ExecutorBaseExecuteModelHook, ModelRunnerExecuteHook, ModelForwardHook, SetForwardContextHook]
