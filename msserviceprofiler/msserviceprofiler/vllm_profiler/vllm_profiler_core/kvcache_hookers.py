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


class KVCacheManagerHook(VLLMHookerBase):
    vllm_version = ("0.6.3", "0.8.4")

    def init(self):
        from vllm.core.block_manager import SelfAttnBlockSpaceManager
        from vllm.engine.llm_engine import LLMEngine

        self.request_dict = {}

        def allocate_maker(ori_func):
            def allocate(this, seq_group, *args, **kwargs):
                profiler = Profiler(Level.INFO)
                ori_func(this, seq_group, *args, **kwargs)
                self.request_dict[seq_group.request_id] = seq_group.seqs
                prof = profiler.domain("KVCache").res(seq_group.request_id)
                prof.metric("deviceBlock", len(this.block_tables)).event("Allocate")

            return allocate

        def append_slots_maker(ori_func):
            def append_slots(this, seq, num_lookahead_slots, *args, **kwargs):
                profiler = Profiler(Level.INFO)
                request_id = seq.seq_id
                for k, v in self.request_dict.items():
                    if seq in v:
                        request_id = k
                        break
                new_cows = ori_func(this, seq, num_lookahead_slots, *args, **kwargs)
                num_blocks = len(getattr(this.block_tables.get(seq.seq_id), "_blocks", []))
                prof = profiler.domain("KVCache").res(request_id)
                prof.metric("deviceBlock", len(this.block_tables)).event("AppendSlot")
                # This is called for every decoder process, recording current blocks
                profiler.domain("KVCache").res(request_id).metric("deviceBlock", num_blocks).event("blocks")
                return new_cows

            return append_slots

        def swap_in_maker(ori_func):
            def swap_in(this, seq_group, *args, **kwargs):
                profiler = Profiler(Level.INFO)
                res = ori_func(this, seq_group, *args, **kwargs)
                prof = profiler.domain("KVCache").res(seq_group.request_id)
                prof.attr("swap", "swap_in").metric("deviceBlock", len(this.block_tables)).event("SwapIn")
                return res

            return swap_in

        def swap_out_maker(ori_func):
            def swap_out(this, seq_group, *args, **kwargs):
                profiler = Profiler(Level.INFO)
                res = ori_func(this, seq_group, *args, **kwargs)
                prof = profiler.domain("KVCache").res(seq_group.request_id)
                prof.attr("swap", "swap_out").metric("deviceBlock", len(this.block_tables)).event("SwapOut")
                return res

            return swap_out

        def free_maker(ori_func):
            def free(this, seq, *args, **kwargs):
                profiler = Profiler(Level.INFO)
                request_id = seq.seq_id
                for k, v in self.request_dict.items():
                    if seq in v:
                        request_id = k
                        break
                ori_func(this, seq, *args, **kwargs)
                profiler.domain("KVCache").res(request_id).metric("deviceBlock", len(this.block_tables)).event("Free")

            return free

        def get_stats_maker(ori_func):
            def get_stats(this, *args, **kwargs):
                profiler = Profiler(Level.INFO)
                stats = ori_func(this, *args, **kwargs)
                num_free_gpu = sum(scheduler.block_manager.get_num_free_gpu_blocks() for scheduler in this.scheduler)
                profiler.domain("KVCache").attr("cpuHitCache", stats.cpu_cache_usage_sys)
                profiler.attr("hitCache", stats.gpu_cache_usage_sys).event("GetCacheHitRate")
                profiler.domain("KVCache").attr("deviceFreeBlock", num_free_gpu).event("GetCacheHitRate")
                return stats

            return get_stats

        self.do_hook([SelfAttnBlockSpaceManager.allocate], allocate_maker)
        self.do_hook([SelfAttnBlockSpaceManager.append_slots], append_slots_maker)
        self.do_hook([SelfAttnBlockSpaceManager.swap_in], swap_in_maker)
        self.do_hook([SelfAttnBlockSpaceManager.swap_out], swap_out_maker)
        self.do_hook([SelfAttnBlockSpaceManager.free], free_maker)
        self.do_hook([LLMEngine._get_stats], get_stats_maker)


kvcache_hookers = [KVCacheManagerHook]
