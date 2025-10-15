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
import importlib.metadata as importlib_metadata

from .utils import logger, set_log_level
from .module_hook import apply_hooks

set_log_level("info")  # Default is info, put here for user changes


def _parse_version_tuple(version_str):
    parts = version_str.split("+")[0].split("-")[0].split(".")
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            break
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def _auto_detect_v1_default():
    """Auto decide default V1 usage based on installed vLLM version.

    Heuristic: for newer vLLM (>= 0.9.2) default to V1, otherwise V0.
    If version can't be determined, fall back to V0 for safety.
    """
    try:
        vllm_version = importlib_metadata.version("vllm")
        major, minor, patch = _parse_version_tuple(vllm_version)
        use_v1 = (major, minor, patch) >= (0, 9, 2)
        logger.info(
            f"VLLM_USE_V1 not set, auto-detected via vLLM {vllm_version}: default {'1' if use_v1 else '0'}"
        )
        return "1" if use_v1 else "0"
    except Exception as e:
        logger.info("VLLM_USE_V1 not set and vLLM version unknown; default to 0 (V0)")
        return "0"


_env_v1 = os.environ.get('VLLM_USE_V1')
VLLM_USE_V1 = _env_v1 if _env_v1 is not None else _auto_detect_v1_default()


def register_service_profiler():
    init_service_profiler()
    

def init_service_profiler():
    if VLLM_USE_V1 == "0":
        from .vllm_v0 import batch_hookers, kvcache_hookers, model_hookers, request_hookers
        apply_hooks()  # 应用所有hookers
    elif VLLM_USE_V1 == "1":
        from .vllm_v1 import batch_hookers, kvcache_hookers, model_hookers, request_hookers
        apply_hooks()  # 应用所有hookers
    else:
        logger.error(f"unknown vLLM interface version: VLLM_USE_V1={VLLM_USE_V1}")
