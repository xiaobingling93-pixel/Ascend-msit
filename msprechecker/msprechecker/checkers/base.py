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

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict

from ..presets import RuleManager
from ..utils import Traverser, MacroExpander, ExpandError, Evaluator, get_handler, ErrorType, CheckError, ErrorHandler
from ..validators import get_validator


class BaseChecker(ABC):
    def __init__(self, *, error_handler: ErrorHandler = None, rule_manager: RuleManager = None):
        self.error_handler = error_handler or get_handler(ErrorType.ERR_CHECK)
        self.rule_manager = rule_manager or RuleManager("")

    @abstractmethod
    def check(self, result: Dict) -> CheckError:
        pass


class NodeChecker(BaseChecker):
    @abstractmethod
    def _get_rules(self):
        pass
    
    def check(self, result: Dict) -> ErrorHandler:
        visited_nodes = Traverser.traverse(result)
        rules = self._get_rules()
        self._validate_nodes(rules, visited_nodes)
        return self.error_handler

    def _validate_nodes(self, rules, visited_nodes):
        queue = deque()
        queue.append((rules, ""))

        while queue:
            node, path = queue.popleft()
            self._dispatch_node(node, path, queue, visited_nodes)

    def _dispatch_node(self, node, path, queue, visited_nodes):
        if isinstance(node, dict) and 'expected' in node:
            self._validate_expected(node, path, visited_nodes)
        elif isinstance(node, dict):
            self._validate_dict(node, path, queue, visited_nodes)
        elif isinstance(node, list):
            self._validate_list(node, path, queue, visited_nodes)

    def _validate_expected(self, node: dict, path: str, visited_nodes: dict):
        expect = node['expected']
        actual = visited_nodes.get(path)
        expected_value = expect['value']

        try:
            expanded_value = MacroExpander.expand(expected_value, path, visited_nodes)
        except ExpandError as e:
            self.error_handler.add_error(
                path=path,
                actual=actual,
                expected=expected_value,
                reason=str(e),
                severity="high"
            )
            return

        validator = get_validator(expect['type'])
        evaluated_value = Evaluator.evaluate(expanded_value)
        reason = node.get('reason', '暂无原因')
        severity = node.get('severity', "high")

        if not validator.validate(actual, evaluated_value):
            self.error_handler.add_error(
                path=path,
                expected=evaluated_value,
                actual=actual,
                reason=reason,
                severity=severity
            )

    def _validate_dict(self, node: dict, path: str, queue: deque, visited_nodes: dict):
        for k, v in node.items():
            new_path = k if not path else f"{path}.{k}"
            if new_path not in visited_nodes:
                self._handle_missing_key(v, new_path)
                continue
            queue.append((v, new_path))

    def _handle_missing_key(self, node, path):
        expected = node.get('expected', {})
        expected_type = expected.get('type', "")
        expected_value = expected.get('value', "")
        if not (expected_type == 'eq' and expected_value is None):
            self.error_handler.add_error(
                path=path,
                actual='missing',
                expected=node.get('expected', {}).get('value', '存在'),
                reason=node.get('reason', '这个字段应该存在'),
                severity=node.get('severity', 'high')
            )

    def _validate_list(self, node: list, path: str, queue: deque, visited_nodes: dict):
        for i, v in enumerate(node):
            new_path = f"[{i}]" if not path else f"{path}[{i}]"
            if new_path not in visited_nodes:
                self._handle_missing_key(v, new_path)
                continue
            queue.append((v, new_path))
