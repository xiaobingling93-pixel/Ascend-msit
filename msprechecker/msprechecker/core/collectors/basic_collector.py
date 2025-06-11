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

import platform
from .sys_collector import AscendInfoCollector

from ..utils.version import get_pkg_version
from .base import BaseCollector


class BasicCollector(BaseCollector):
    def __init__(self, args) -> None:
        self.args = args

    def collect(self) -> dict:
        # Platform info
        platform_info = f"{platform.platform()} -- Python {platform.python_version()}"
        transformers_ver = get_pkg_version("transformers")
        torch_ver = get_pkg_version("torch")
        torch_npu_ver = get_pkg_version("torch_npu")
        platform_info += f", torch {torch_ver}, torch_npu {torch_npu_ver}, transformers {transformers_ver}"

        # Ascend info (toolkit, mindie, atb, atb-models)
        ascend_info = AscendInfoCollector().collect() or {}
        driver_version = ascend_info.get("driver", {}).get("version", "not found")
        toolkit_version = ascend_info.get("toolkit", {}).get("version", "not found")
        toolkit_time = ascend_info.get("toolkit", {}).get("time", "not found")
        mindie_version = ascend_info.get("mindie", {}).get("version", "not found")
        atb_version = ascend_info.get("atb", {}).get("version", "unknown")
        atb_models_version = ascend_info.get("atb-models", {}).get("version", "unknown")
        atb_models_time = ascend_info.get("atb-models", {}).get("time", "unknown")

        return {
            "platform": platform_info,
            "driver": driver_version,
            "toolkit": f"{toolkit_version} -- {toolkit_time}",
            "mindie": mindie_version,
            "atb": f"{atb_version}",
            "atb-models": f"{atb_models_version} -- {atb_models_time}",
        }
