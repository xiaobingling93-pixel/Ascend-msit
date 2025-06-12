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
import unittest
from unittest.mock import patch, MagicMock


class TestInit(unittest.TestCase):
    @patch('msserviceprofiler.vllm_profiler.vllm_profiler_core.model_hookers.model_hookers', [])
    @patch('msserviceprofiler.vllm_profiler.vllm_profiler_core.batch_hookers.batch_hookers', [])
    @patch('msserviceprofiler.vllm_profiler.vllm_profiler_core.kvcache_hookers.kvcache_hookers', [])
    @patch('msserviceprofiler.vllm_profiler.vllm_profiler_core.request_hookers.request_hookers', [])
    def test_empty_hookers(self, *args):
        """测试空hooker列表情况"""
        from msserviceprofiler.vllm_profiler.vllm_profiler_0_6_3 import all_hookers
        self.assertEqual(all_hookers, [])

    def test_single_hooker(self):
        """测试单个hooker初始化"""
        mock_hooker = MagicMock()
        mock_hooker.return_value.support_version.return_value = True
        
        with patch('msserviceprofiler.vllm_profiler.vllm_profiler_core.model_hookers.model_hookers', [mock_hooker]), \
             patch('msserviceprofiler.vllm_profiler.vllm_profiler_core.batch_hookers.batch_hookers', []), \
             patch('msserviceprofiler.vllm_profiler.vllm_profiler_core.kvcache_hookers.kvcache_hookers', []), \
             patch('msserviceprofiler.vllm_profiler.vllm_profiler_core.request_hookers.request_hookers', []):
            
            # 必须重新import才能获取patch后的值
            import importlib
            import msserviceprofiler.vllm_profiler.vllm_profiler_core
            importlib.reload(msserviceprofiler.vllm_profiler.vllm_profiler_core)
            
            from msserviceprofiler.vllm_profiler.vllm_profiler_0_6_3 import all_hookers
            self.assertEqual(len(all_hookers), 0)