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
from msserviceprofiler.vllm_profiler.vllm_profiler_core.model_hookers import model_hookers
from msserviceprofiler.vllm_profiler.vllm_profiler_core.batch_hookers import batch_hookers
from msserviceprofiler.vllm_profiler.vllm_profiler_core.kvcache_hookers import kvcache_hookers
from msserviceprofiler.vllm_profiler.vllm_profiler_core.request_hookers import request_hookers

all_hookers = kvcache_hookers
all_hookers += model_hookers
all_hookers += batch_hookers
all_hookers += request_hookers

for hook_cls in all_hookers:
    hooker = hook_cls()
    if hooker.support_version("0.6.3"):
        hooker.init()
