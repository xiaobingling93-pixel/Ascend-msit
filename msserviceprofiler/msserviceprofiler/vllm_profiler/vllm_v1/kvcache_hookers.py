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
from collections import Counter
from ms_service_profiler import Profiler, Level
from ..module_hook import vllm_hook

@vllm_hook(("vllm.v1.core.kv_cache_manager", "KVCacheManager.allocate_slots"), min_version="0.9.1")
def allocate_slots(original_func, this, request, *args, **kwargs):
    ret = original_func(this, request, *args, **kwargs)
    prof = Profiler(Level.INFO).domain("KVCache").res(request.request_id)
    prof.metric("deviceBlock", len(this.block_tables)).event("Allocate")
    print(f">>>> {this.usage()}")
    return ret
