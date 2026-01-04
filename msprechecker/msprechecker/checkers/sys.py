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

import subprocess
from .base import NodeChecker
from ..utils import Traverser, get_conn_mode


class SysChecker(NodeChecker):
    def __init__(self, *, error_handler=None, rule_manager=None):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.error_handler.type = "system"

    def _get_rules(self):
        if self.rule_manager.framework == "vllm" or self.rule_manager.scene == "vllm":
            self.rule_manager.scene == "default"
            return self.rule_manager.get_rules().get("sys")
        self.rule_manager.scene = "default"
        return self.rule_manager.get_rules().get("sys")

    def _check(self, results) -> None:
        # 先执行原有的节点检查
        visited_nodes = Traverser.traverse(results)
        rules = self._get_rules()
        self._validate_nodes(rules, visited_nodes)

        self._check_connected_route()
        
        # 在vllm框架下额外检查jemalloc安装
        if self.rule_manager.framework == "vllm":
            self._check_jemalloc_installation()

    def _check_jemalloc_installation(self):
        """检查jemalloc是否通过包管理器安装"""
        jemalloc_installed = self._is_jemalloc_installed()
        
        if not jemalloc_installed:
            self.error_handler.add_error(
                path="jemalloc",
                actual="not installed",
                expected="installed",
                reason="jemalloc未通过apt/yum安装，建议安装以提高性能: "
                       "Ubuntu/Debian: sudo apt-get install libjemalloc-dev, "
                       "CentOS/RHEL: sudo yum install jemalloc",
                severity="low"
            )

    def _is_jemalloc_installed(self) -> bool:
        """检查jemalloc是否通过包管理器安装"""
        # 先尝试Ubuntu/Debian的apt
        try:
            result_apt = subprocess.run(
                ['/usr/bin/apt', 'list', '--installed', 'libjemalloc*'],
                capture_output=True,
                text=True,
                check=False
            )
            if result_apt.returncode == 0 and 'libjemalloc' in result_apt.stdout:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # 再尝试CentOS/RHEL的yum
        try:
            result_yum = subprocess.run(
                ['/usr/bin/yum', 'list', 'installed', 'jemalloc*'],
                capture_output=True,
                text=True,
                check=False
            )
            if result_yum.returncode == 0 and 'jemalloc' in result_yum.stdout:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return False

    def _check_connected_route(self):
        connect_mode = get_conn_mode()
        warning_msg = (
            "检测到网线对端设备为昇腾 NPU。"
            '请确认当前部署环境是否为“双机背靠背直连”（即双机一体机）架构。'
            '此架构下，HCCL 不支持全互联通信链路的自动建立，模型通信可能会受到影响。'
        )
        if connect_mode in {'fiber', None}:
            self.error_handler.add_error(
                path="HCCL 链路层协议",
                actual=connect_mode or '未知协议',
                expected="-",
                reason=warning_msg,
                severity="medium"
            )
