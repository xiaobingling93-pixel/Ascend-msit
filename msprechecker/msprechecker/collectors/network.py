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
from ..utils import get_current_ip_and_addr, get_rank_table_parser


class PingCollector(BaseCollector):
    PING_COMMAND_TERMPLATE = "ping -c 3 -q -W 2 {}"
    PING_RESULT_PATTERN = re.compile(
        r"rtt min/avg/max/mdev = (?P<min>\d+\.\d+)/(?P<avg>\d+\.\d+)/(?P<max>\d+\.\d+)/(?P<mdev>\d+\.\d+) ms"
    )

    def __init__(self, error_handler=None, *, rank_table_file=None, rank_table_parser=None):
        super().__init__(error_handler)
        self.error_handler.type = "ping"
        self.rank_table_file = rank_table_file
        self.rank_table_parser = rank_table_parser or get_rank_table_parser()

    def _collect_data(self):
        if not self._check_ping_command():
            return {}

        if not self._check_rank_table_file():
            return {}

        cur_ip = self._get_current_ip()

        ip_to_rank_id = self._parse_rank_table()
        if not ip_to_rank_id:
            return {}

        if cur_ip in ip_to_rank_id:
            ip_to_rank_id.pop(cur_ip)

        return self._ping_all_ips(ip_to_rank_id)

    def _check_ping_command(self):
        if shutil.which("ping") is None:
            self.error_handler.add_error(
                filename=__file__,
                function='_check_ping_command',
                lineno=56,
                what="当前环境没有 'ping' 命令",
                reason="需要使用 'ping' 命令来进行多机的连通性检测"
            )
            return False
        return True

    def _check_rank_table_file(self):
        if not self.rank_table_file:
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data',
                lineno=45,
                what="未传入 'rank table'",
                reason="需要使用 'rank table' 获取所有机器 IP"
            )
            return False
        return True

    def _get_current_ip(self):
        _, current_ip = get_current_ip_and_addr()
        if not current_ip:
            self.error_handler.add_error(
                filename=__file__,
                function='_get_current_ip',
                lineno=81,
                what="获取当前机器 IP 失败",
                reason="需要获取当前 IP 从而获取其他机器的 device ip"
            )
            return ""
        return current_ip

    def _parse_rank_table(self):
        try:
            parser = self.rank_table_parser(self.rank_table_file)
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function='_parse_rank_table',
                lineno=94,
                what="尝试读取传入的 'rank table' 失败",
                reason=str(e)
            )
            return {}
        try:
            ip_to_rank_id = parser.parse()
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function='_parse_rank_table',
                lineno=105,
                what="尝试解析传入的 'rank table' 失败",
                reason=str(e)
            )
            return {}
        if not ip_to_rank_id:
            self.error_handler.add_error(
                filename=__file__,
                function='_parse_rank_table',
                lineno=115,
                what="传入的 'rank table' 没有解析出任何信息",
                reason="请检查 'rank table' 是否符合格式规范"
            )
            return {}
        return ip_to_rank_id

    def _ping_all_ips(self, ip_to_rank_id):
        ret = {}
        for remote_ip in ip_to_rank_id:
            output = self._ping_ip(remote_ip)
            avg = self._parse_ping_output(output)
            ret[remote_ip] = avg if avg is not None else output
        return ret

    def _ping_ip(self, remote_ip):
        try:
            output = subprocess.check_output(
                shlex.split(self.PING_COMMAND_TERMPLATE.format(remote_ip)),
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5
            )
            return output
        except Exception as e:
            return str(e)

    def _parse_ping_output(self, output):
        m = self.PING_RESULT_PATTERN.search(output)
        if m:
            return m.group("avg")
        return None
