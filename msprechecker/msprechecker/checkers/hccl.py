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

from .base import NodeChecker
from ..utils import get_npu_count


class HCCLChecker(NodeChecker):
    def __init__(self, *, error_handler=None, rule_manager=None):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.error_handler.type = "hccl"

    def _get_rules(self):
        npu_count = get_npu_count()

        if not npu_count:
            return {}

        return {
            'gateway': self.rule_manager.create_rule(
                'ne',
                ['null' for _ in range(npu_count)],
                '当前的机器的 RoCE 网卡默认网关需要配置，如果是光纤直连请忽略此项',
                'low'
            ),
            'ip': self.rule_manager.create_rule(
                'ne',
                ['null' for _ in range(npu_count)],
                '当前机器的 RoCE 网卡 IP 需要配置在同一网段',
                'high'
            ),
            'link': self.rule_manager.create_rule(
                'eq',
                ['UP' for _ in range(npu_count)],
                '当前机器的网卡 LINK 状态应该为全 "UP"',
                'high'
            ),
            'net_health': self.rule_manager.create_rule(
                'eq',
                ['Success' for _ in range(npu_count)],
                '当前机器的 RoCE 网卡与配置对端 IP 连通应该都为 "Success"',
                'low'
            ),
            'netdetect': self.rule_manager.create_rule(
                'ne',
                ['null' for _ in range(npu_count)],
                '当前机器的 RoCE 网卡对端 IP 需要配置',
                'low'
            ),
            'tls': self.rule_manager.create_rule(
                'eq',
                ['0' for _ in range(npu_count)],
                '当前机器的 TLS 证书需要被使能，需要在 root 用户下执行该项检测',
                'low'
            )
        }
