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

from .vllm_hooker_base import VLLMHookerBase
from ms_service_profiler import Profiler, Level


class ModelRunnerExecuteHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.6.3")

    def init(self):
        from vllm.worker.model_runner import ModelRunner

        def execute_model_maker(ori_func):
            def execute_model(this, model_input, kv_caches, *args, **kwargs):
                prof = Profiler(Level.INFO)
                prof.span_start("ModelExec")
                ret = ori_func(this, model_input, kv_caches, *args, **kwargs)

                is_prefill = model_input.attn_metadata.prefill_metadata

                request_id_list = []

                for request_id, _ in model_input.request_ids_to_seq_ids.items():
                    request_id_list.append(request_id)

                prof.res(request_id_list)

                if is_prefill:
                    prof.attr("batch_type", "prefill")
                else:
                    prof.attr("batch_type", "decode")

                batch_size = model_input.input_tokens.shape[0]
                prof.attr("batch_size", batch_size)

                prof.span_end()
                return ret

            return execute_model

        sefl.do_hook([ModelRunner.execute_model], execute_model_maker)


model_hookers = [ModelRunnerExecuteHook]
