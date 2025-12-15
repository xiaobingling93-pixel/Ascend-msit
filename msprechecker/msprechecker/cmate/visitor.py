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
import os
import operator
from collections import defaultdict

from msguard.security import open_s

from . import _ast
from .util import Severity, cmate_logger
from .data_source import NA


class CMateError(Exception):
    pass


class NodeVisitor:
    def visit(self, node):
        if node is None:
            return None

        method_name = f"visit_{node.__class__.__name__.lower()}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if node is None or not hasattr(node, '__slots__'):
            return

        for attr in node.__slots__:
            val = getattr(node, attr, None)

            if isinstance(val, _ast.Node):
                self.visit(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, _ast.Node):
                        self.visit(item)


class MetaVisitor(NodeVisitor):
    def __init__(self):
        self._meta_data = {}

    def visit_meta(self, node: _ast.Meta):
        if self._meta_data:
            raise CMateError('Multiple meta declarations found')

        for assign_node in node.body:
            target_node = assign_node.target
            value_node = assign_node.value
            self._meta_data[target_node.id] = self._retrieve_value(value_node)

    def visit(self, node):
        super().visit(node)

        return self._meta_data

    def _retrieve_value(self, node):
        if node is None:
            return None

        if isinstance(node, _ast.Constant):
            return node.value

        if isinstance(node, _ast.List):
            return [self._retrieve_value(elt) for elt in node.elts]

        if isinstance(node, _ast.Dict):
            keys = [self._retrieve_value(key) for key in node.keys]
            values = [self._retrieve_value(value) for value in node.values]
            return dict(zip(keys, values))

        raise CMateError(f'Unsupported node type: {type(node).__name__}')


class RequirementGenerator(NodeVisitor):
    def __init__(self):
        self._current_namespace = None
        self._partition_map = {}
        self._dependency_map = {}

    def visit_meta(self, node):
        pass

    def visit_global(self, node):
        pass

    def visit_dependency(self, node: _ast.Dependency):
        if self._dependency_map:
            raise CMateError('Multiple dependency declarations found')

        for desc_node in node.body:
            self._dependency_map[desc_node.target.id] = (desc_node.desc, desc_node.parse_type)

    def visit_partition(self, node: _ast.Partition):
        self._current_namespace = node.target.id
        if self._current_namespace in self._partition_map:
            raise CMateError
        
        self._partition_map[self._current_namespace] = defaultdict(set)

        for rule_node in node.body:
            self.visit(rule_node)
        
        self._current_namespace = None
    
    def visit_compare(self, node: _ast.Compare):
        if (isinstance(node.left, _ast.DictPath) and 
            node.left.namespace == 'context' and 
            isinstance(node.comparator, _ast.Constant)):

            possible_val = node.comparator.value
            self._partition_map[self._current_namespace][node.left.path].add(possible_val)

        self.visit(node.left)
        self.visit(node.comparator)

    def generate(self, node):
        self.visit(node)
        
        res = {}
        for namespace in self._partition_map:
            contexts = {}
            desc, parse_type = self._dependency_map.get(namespace, (None, None))

            res[namespace] = {
                'desc': desc,
                'parse_type': parse_type,
                'contexts': contexts
            }
            for name, possible_values in self._partition_map[namespace].items():
                desc = self._dependency_map.get(name)
                contexts[name] = {
                    'desc': desc,
                    'possible_values': list(possible_values)
                }

        return res


class Evaluator(NodeVisitor):
    def __init__(self, data_source):
        self.data_source = data_source
        self.op_map = {
            '==': operator.eq,
            '<=': operator.le,
            '>=': operator.gt,
            '!=': operator.ne,
            'or': lambda a, b: a or b,
            'in': lambda a, b: a in b,
            'and': lambda a, b: a and b,
            '*': operator.mul,
            '+': operator.add
        }
        self.func_map = {
            'int': int,
            'len': len,
            'range': range,
            'PathExists': lambda a: a is not None and a is not NA and os.path.exists(a)
        }
        self.history = None
    
    def visit_name(self, node: _ast.Name):
        path = f'global::{node.id}'
        return self.data_source[path]

    def visit_constant(self, node: _ast.Constant):
        return node.value
    
    def visit_dictpath(self, node: _ast.DictPath):
        path = f'{node.namespace}::{node.path}'
        val = self.data_source[path]
        
        self.history.append((path, val))
        return val[0] if isinstance(val, tuple) else val
    
    def visit_list(self, node: _ast.List):
        return [self.visit(elt) for elt in node.elts]
    
    def visit_dict(self, node: _ast.Dict):
        keys = [self.visit(key) for key in node.keys]
        values = [self.visit(value) for value in node.values]

        return dict(zip(keys, values))
    
    def visit_compare(self, node: _ast.Compare):
        op = self.op_map[node.op]
        left = self.visit(node.left)
        comparator = self.visit(node.comparator)

        return op(left, comparator)
    
    def visit_call(self, node: _ast.Call):
        func = self.func_map[node.func.id]
        args = [self.visit(arg) for arg in node.args]

        return func(*args)
    
    def visit_unaryop(self, node: _ast.UnaryOp):
        op = self.op_map[node.op]
        operand = self.visit(node.operand)

        return op(operand)
    
    def visit_binop(self, node: _ast.BinOp):
        op = self.op_map[node.op]
        left = self.visit(node.left)
        right = self.visit(node.right)

        return op(left, right)
    
    def eval(self, node: _ast.Node):
        self.history = []
        return self.visit(node)


class PrettyFormatter(NodeVisitor):
    def visit_name(self, node: _ast.Name):
        return node.id

    def visit_constant(self, node: _ast.Constant):
        return node.value
    
    def visit_dictpath(self, node: _ast.DictPath):
        return f'{node.namespace}::{node.path}'
    
    def visit_list(self, node: _ast.List):
        return [self.visit(elt) for elt in node.elts]
    
    def visit_dict(self, node: _ast.Dict):
        keys = [self.visit(key) for key in node.keys]
        values = [self.visit(value) for value in node.values]

        return dict(zip(keys, values))

    def visit_compare(self, node: _ast.Compare):
        left = self.visit(node.left)
        comparator = self.visit(node.comparator)

        return f'{left} {node.op} {comparator}'
    
    def visit_call(self, node: _ast.Call):
        args = [self.visit(arg) for arg in node.args]
        return f'{node.func.id}{tuple(args)}'
    
    def visit_unaryop(self, node: _ast.UnaryOp):
        operand = self.visit(node.operand)

        return f'{node.op} {operand}'
    
    def visit_binop(self, node: _ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)

        return f'{left} {node.op} {right}'
    
    def visit_rule(self, node: _ast.Rule):
        return self.visit(node.test)
    
    def format(self, node: _ast.Node):
        return self.visit(node)


class RuleVisitor(NodeVisitor):
    def __init__(self, data_source, severity):
        self._data_source = data_source
        self._severity = Severity[severity.upper()]

        self._pretty_formatter = PrettyFormatter()
        self._evaluator = Evaluator(self._data_source)

        self._section = None
        self._current_namespace = None
        self._ruleset = defaultdict(set)
    
    def visit_meta(self, node):
        pass

    def visit_dependency(self, node):
        pass
    
    def visit_global(self, node: _ast.Global):
        self._section = 'global'

        for assign_node in node.body:
            self.visit(assign_node)
        
        self._section = None
    
    def visit_partition(self, node: _ast.Partition):
        self._section = 'partition'
        self._current_namespace = node.target.id

        for rule_node in node.body:
            self.visit(rule_node)

        self._current_namespace = None
        self._section = None

    def visit_for(self, node: _ast.For):
        # evaluate iterable in current local context
        try:
            evaluator = Evaluator(self._data_source, local_vars=self._merged_locals(),
                                      default_namespace=getattr(self._current_partition.target, 'id', None))
            iter_val = evaluator.eval(node.it)
        except Exception:
            iter_val = None

        if iter_val is None:
            return

        # iterate and visit body for each binding
        for val in iter_val:
            self._local_stack.append({node.target.id: val})
            for stmt in node.body:
                self.visit(stmt)
            self._local_stack.pop()

    def visit_if(self, node: _ast.If):
        test = node.test
        self.visit(test)

        if self._evaluator.eval(test):
            for stmt in node.body:
                self.visit(stmt)
        elif node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_assign(self, node: _ast.Assign):
        target = node.target.id
    
        self.visit(node.value) # visit first

        path = f'global::{target}'
        self._data_source[path] = (
            self._evaluator.eval(node.value),
            self._pretty_formatter.format(node.value)
        )

    def visit_rule(self, node: _ast.Rule):
        if self._severity > node.severity:
            return

        self.visit(node.test) # visit first
        self._ruleset[self._current_namespace].add(node)

    def visit_dictpath(self, node: _ast.DictPath):
        if self._section == 'global' and node.namespace is None:
            raise RuntimeError

        if node.namespace is None:
            local_path = f'{self._current_namespace}::{node.path}'
            global_path = f'global::{node.path}'

            if local_path not in self._data_source and global_path in self._data_source:
                node.namespace = 'global'
            else:
                node.namespace = self._current_namespace
    
    def visit(self, node: _ast):
        super().visit(node)

        return self._ruleset


class SetEnvGenerator(NodeVisitor):
    def __init__(self, data_source):
        self._evaluator = Evaluator(data_source)
        self._white_list = re.compile(r'[a-zA-Z0-9_\-:.]+')
        self._environ = os.environ.copy()
        self._set_scripts = []
        self._undo_scripts = []

    def visit_meta(self, node):
        pass

    def visit_dependency(self, node):
        pass

    def visit_global(self, node):
        pass

    def visit_partition(self, node: _ast.Partition):
        namespace = node.target.id

        if not namespace == 'env':
            return
        
        for rule_node in node.body:
            self.visit(rule_node)
    
    def visit_compare(self, node: _ast.Compare):
        if not isinstance(node.left, _ast.DictPath) or node.left.namespace != 'env':
            return

        env_var = node.left.path

        if node.op == '==':
            expected_val = self._evaluator.eval(node.comparator)
        elif node.op == 'in':
            expected_val = self._evaluator.eval(node.comparator)[0]
        else:
            return

        if expected_val is NA:
            self._set_scripts.append(f'unset {env_var}')
            return

        if expected_val is None:
            cmate_logger.warning(
                "At line %d, column %d: 'None' cannot be used to unset environment variables. "
                "Please use the special value 'NA' to explicitly leave a variable unset.",
                node.lineno, node.col_offset
            )
            self._set_scripts.append(f'unset {env_var}')
            return

        if not isinstance(expected_val, str):
            cmate_logger.warning(
                "At line %d, column %d: non-str value cannot be used to set environment variables.",
                node.lineno, node.col_offset
            )
            return

        if not self._white_list.fullmatch(expected_val):
            cmate_logger.warning(
                "At line %d, column %d: Unexpected value '%s' detected. "
                "This value is not in the allowed format/pattern.",
                node.lineno, node.col_offset, expected_val
            )
            return

        original_val = self._environ.get(env_var)

        self._set_scripts.append(f'export {env_var}={expected_val}')
        if original_val is None:
            self._undo_scripts.append(f'unset {env_var}')
        else:
            self._undo_scripts.append(f'export {env_var}={original_val}')
    
    def generate(self, node):
        self.visit(node)

        script_format = (
            '#!/bin/bash\n'
            '# Environment variable management script auto-generated by cmate\n'
            '# Usage:\n'
            '#   source set_env.sh    # Apply environment changes\n'
            '#   source set_env.sh 0  # Revert changes\n\n'
            'if [ "$1" = "0" ]; then\n'
            "    {set}\n"
            "else\n"
            "    {undo}\n"
            "fi\n"
        )

        with open_s('set_env.sh', 'w') as f:
            script = script_format.format(
                set='\n    '.join(self._set_scripts),
                undo='\n    '.join(self._undo_scripts)
            )
            f.write(script)
