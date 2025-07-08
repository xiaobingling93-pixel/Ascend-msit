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
        driver_version = ascend_info.get("driver", {}).get("version", "not installed")
        toolkit_version = ascend_info.get("toolkit", {}).get("version", "not installed")
        toolkit_time = ascend_info.get("toolkit", {}).get("time", "not installed")
        opp_kernel_version = ascend_info.get("opp_kernel", {}).get("version", "not installed")
        opp_kernel_time = ascend_info.get("opp_kernel", {}).get("time", "not installed")
        mindie_version = ascend_info.get("mindie", {}).get("version", "not installed")
        atb_version = ascend_info.get("atb", {}).get("version", "not installed")
        atb_models_version = ascend_info.get("atb-models", {}).get("version", "not installed")
        atb_models_time = ascend_info.get("atb-models", {}).get("time", "not installed")

        return {
            "platform": platform_info,
            "driver": driver_version,
            "toolkit": f"{toolkit_version} -- {toolkit_time}" if toolkit_version else "not installed",
            "opp_kernel": f"{opp_kernel_version} -- {opp_kernel_time}" if opp_kernel_version else "not installed",
            "mindie": mindie_version or "not installed",
            "atb": atb_version or "not installed",
            "atb-models": f"{atb_models_version} -- {atb_models_time}" if atb_models_version else "not installed"
        }
