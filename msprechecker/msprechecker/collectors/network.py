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

import re
import shlex
import shutil
import subprocess

from .base import BaseCollector
from ..utils import get_current_ip_and_addr


class PingCollector(BaseCollector):
    def __init__(self, error_handler=None, *, rank_table=None):
        super().__init__(error_handler)
        self.rank_table = rank_table
        
        ping_cmd = "/usr/bin/ping"
        self._ping_cmd = ping_cmd + " -c 3 -q -W 2 {}" if shutil.which(ping_cmd) else None

    def _collect_data(self):
        result = {}

        if not self._ping_cmd:
            self.error_handler.add_error(
                filename=__file__, function='_collect_data',
                lineno=34, what="当前环境没有 'ping' 命令",
                reason="需要使用 'ping' 命令来进行多机的连通性检测"
            )
            return result

        host_to_devices = self.rank_table.host_to_devices
        if not host_to_devices:
            self.error_handler.add_error(
                filename=__file__, function='_collect_data',
                lineno=45, what="传入的 'rank table' 没有解析出任何信息",
                reason="请检查 'rank table' 是否符合格式规范"
            )
            return result

        _, cur_ip = get_current_ip_and_addr()
        if not cur_ip:
            self.error_handler.add_error(
                filename=__file__, function='get_current_ip_and_addr',
                lineno=55, what="获取当前机器 IP 失败",
                reason="需要获取当前 IP 从而获取其他机器的 ip"
            )
            return result
        
        result = {}
        for host in host_to_devices:
            try:
                output = subprocess.check_output(
                    shlex.split(self._ping_cmd.format(host)),
                    stderr=subprocess.STDOUT, text=True, timeout=5
                )
            except Exception as e:
                self.error_handler.add_error(
                    filename=__file__, function='subprocess.check_output',
                    lineno=55, what=f"执行命令失败：{self._ping_cmd.format(host)}",
                    reason=str(e)
                )
                output = ""
            result[host] = output

        return result
