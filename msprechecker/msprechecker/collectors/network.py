# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import shlex
import shutil
import subprocess

from msguard import Rule

from .base import BaseCollector


class PingCollector(BaseCollector):
    def __init__(self, error_handler=None, *, rank_table=None):
        super().__init__(error_handler)
        self.rank_table = rank_table

        ping_cmd = "/usr/bin/ping"
        if not Rule.input_file_exec.is_satisfied_by(ping_cmd):
            self._ping_cmd = None
        else:
            self._ping_cmd = ping_cmd + " -c 3 -q -W 2 {}"

    def _collect_data(self):
        result = {}

        if not self._ping_cmd:
            self.error_handler.add_error(
                filename=__file__, function='_collect_data',
                lineno=36, what="当前环境没有 'ping' 命令或者权限不符合要求",
                reason="需要使用 'ping' 命令来进行多机的连通性检测"
            )
            return result

        host_to_devices = self.rank_table.host_to_devices
        if not host_to_devices:
            self.error_handler.add_error(
                filename=__file__, function='_collect_data',
                lineno=47, what="传入的 'rank table' 没有解析出任何信息",
                reason="请检查 'rank table' 是否符合格式规范"
            )
            return result

        for host in host_to_devices:
            try:
                output = subprocess.check_output(
                    shlex.split(self._ping_cmd.format(host)),
                    stderr=subprocess.STDOUT, text=True, timeout=5
                )
            except Exception:
                output = "ping failed"
            result[host] = output

        return result
