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

from ..base import NodeChecker
from ...utils import get_pkg_version


class ModelConfigChecker(NodeChecker):
    def __init__(self, *, error_handler=None, rule_manager=None):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.error_handler.type = "model config"

    def _get_rules(self):
        cur_transformers_version = get_pkg_version('transformers')

        return {
            "torch_dtype": self.rule_manager.create_rule(
                type_='eq',
                value='float16',
                reason='部分模型的算子可能不支持 bfloat16, 请确保当前模型算子支持 bfloat16',
                severity='low'
            ),
            'transformers_version': self.rule_manager.create_rule(
                type_='le',
                value=cur_transformers_version,
                reason=f'当前机器的 "transformers" 的版本（{cur_transformers_version}）如果小于配置文件要求版本，会导致服务启动失败',
                severity='high'
            )
        }
