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
from importlib.metadata import version
from packaging.version import Version

try:
    vllm_version = version("vllm")
except Exception as ee:
    vllm_version = None


if vllm_version and Version(vllm_version) > Version("0.8.3"):
    import msserviceprofiler.vllm_profiler.vllm_profiler_0_8_4
elif vllm_version and Version(vllm_version) > Version("0.6.2"):
    import msserviceprofiler.vllm_profiler.vllm_profiler_0_6_3
else:
    import logging

    logging.error(f"Not supported vllm version {vllm_version}")
