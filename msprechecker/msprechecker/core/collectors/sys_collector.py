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

from .base import BaseCollector, ParallelCollector
from ..utils.file_utils import read_file_lines


class DriverVersionCollector(BaseCollector):
    version_file = '/usr/local/Ascend/driver/version.info'

    def collect(self):
        lines = read_file_lines(self.version_file)
        if not lines:
            return {"version": None}
        for line in lines:
            if line.startswith('Version='):
                return {"version": line.strip().split('=', 1)[1]}
        return {"version": None}


class ToolkitVersionCollector(BaseCollector):
    default_home = '/usr/local/Ascend/ascend-toolkit/latest'

    def collect(self):
        toolkit_info = {"version": None, "time": None}
        toolkit_home = os.getenv("ASCEND_TOOLKIT_HOME") or self.default_home
        version_file = os.path.join(toolkit_home, "version.cfg")
        lines = read_file_lines(version_file)
        if lines:
            for line in lines:
                if "=" in line and ":" in line:
                    parts = line.replace("=", ":").split(":")
                    if len(parts) > 1:
                        toolkit_info["version"] = parts[-1].strip("]\n ")
                        break
        compiler_version_file = os.path.join(toolkit_home, "compiler", "version.info")
        lines = read_file_lines(compiler_version_file)
        if lines:
            for line in lines:
                if "timestamp=" in line:
                    toolkit_info["time"] = line.split("=", 1)[-1].strip()
                    break
        return toolkit_info


class MindIEVersionCollector(BaseCollector):
    default_home = '/usr/local/Ascend/mindie/latest/mindie-llm'

    def collect(self):
        mindie_info = {"version": None}
        mindie_home = os.getenv("MINDIE_LLM_HOME_PATH") or self.default_home
        version_file = os.path.join(mindie_home, "..", "version.info")
        lines = read_file_lines(version_file)
        if lines:
            for line in lines:
                if "mindie" in line and ':' in line:
                    _, value = line.split(":", 1)
                    mindie_info["version"] = value.strip()
                    break
        return mindie_info


class ATBVersionCollector(BaseCollector):
    default_home = '/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0'
    desired_columns = ["version", "branch", "commit"]

    def collect(self):
        atb_info = {col: None for col in self.desired_columns}
        atb_home = os.getenv("ATB_HOME_PATH") or self.default_home
        version_file = os.path.join(atb_home, "..", "..", "version.info")
        lines = read_file_lines(version_file)
        if lines:
            for line in lines:
                for col in self.desired_columns:
                    if col in line.lower() and ':' in line:
                        _, value = line.split(":", 1)
                        atb_info[col] = value.strip()
        return atb_info


class ATBSpeedVersionCollector(BaseCollector):
    default_home = '/usr/local/Ascend/atb-models'
    desired_columns = ["version", "branch", "commit", "time"]

    def collect(self):
        atb_speed_info = {col: None for col in self.desired_columns}
        atb_speed_home = os.getenv("ATB_SPEED_HOME_PATH") or self.default_home
        version_file = os.path.join(atb_speed_home, "version.info")
        lines = read_file_lines(version_file)
        if lines:
            for line in lines:
                for col in self.desired_columns:
                    if col in line.lower() and ':' in line:
                        _, value = line.split(":", 1)
                        atb_speed_info[col] = value.strip()
        return atb_speed_info


class AscendInfoCollector(ParallelCollector):
    """Collects all Ascend component versions in parallel."""
    def __init__(self):
        super().__init__({
            "toolkit": ToolkitVersionCollector(),
            "atb": ATBVersionCollector(),
            "mindie": MindIEVersionCollector(),
            "atb-models": ATBSpeedVersionCollector(),
            "driver": DriverVersionCollector(),
        })
