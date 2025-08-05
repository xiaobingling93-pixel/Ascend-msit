# -*- coding: utf-8 -*-
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
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from .base import BaseCollector
from ..utils import get_npu_count, is_in_container, get_conn_mode


class HCCLCollector(BaseCollector):
    COMMON_CMD_TEMPLATE = "hccn_tool -i {device} {option} -g"
    HCCN_OPTIONS_TO_REGEX = {
        "ip": re.compile(r"ipaddr:([\d.]+)", re.IGNORECASE),
        "netdetect": re.compile(r"netdetect address:\s*([\d.]+)", re.IGNORECASE),
        "net_health": re.compile(r"net health status:\s*([\w ]+)", re.IGNORECASE),
        "gateway": re.compile(r"gateway:\s*([\d.]+)", re.IGNORECASE),
        "lldp": re.compile(r"Ifname: ([\w/:.]+)", re.IGNORECASE),
        "link": re.compile(r"link status: (\w+)", re.IGNORECASE),
        "tls": re.compile(r"tls switch\[(\d)\]", re.IGNORECASE)
    }

    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        self.error_handler.type = "hccl"

    def _run_hccn_cmd(self, option: str, device: int):
        """
        Run hccn_tool command for a specific option and device, and extract result using regex.
        """
        regex = self.HCCN_OPTIONS_TO_REGEX[option]
        command = self.COMMON_CMD_TEMPLATE.format(device=device, option="-" + option)
        try:
            output = subprocess.check_output(
                shlex.split(command), stderr=subprocess.STDOUT, text=True, timeout=2
            )
        except Exception:
            return None

        match = regex.search(output)
        return match.group(1) if match else None

    def _collect_data(self):
        """
        Collect HCCL related data for all NPUs and options.
        """
        if not shutil.which('hccn_tool'):
            working_place = "宿主机" if not is_in_container() else "容器"
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data',
                lineno=63,
                what=f"{working_place}上没有找到 'hccn_tool' 命令",
                reason="[Errno 2] No such file or directory: 'hccn_tool'"
            )
            return {}

        npu_count = get_npu_count()
        max_workers = min(npu_count * len(self.HCCN_OPTIONS_TO_REGEX), os.cpu_count() or 1)

        metric = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for option in self.HCCN_OPTIONS_TO_REGEX:
                futures = [
                    executor.submit(self._run_hccn_cmd, option, device)
                    for device in range(npu_count)
                ]
                metric[option] = [future.result() for future in futures]

        metric['conn'] = get_conn_mode()
        return metric
