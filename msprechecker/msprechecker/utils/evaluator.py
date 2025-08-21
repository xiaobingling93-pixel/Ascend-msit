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
import operator
from typing import Any

from .version import Version


class Evaluator:
    OPS = {
         # 算术运算符
        '**': (5, operator.pow),
        '*': (4, operator.mul),
        '/': (4, operator.truediv),
        '//': (4, operator.floordiv),
        '%': (4, operator.mod),

        '+': (3, operator.add),
        '-': (3, operator.sub),
        
        # 比较运算符
        '==': (2, operator.eq),
        '!=': (2, operator.ne),
        '>': (2, operator.gt),
        '<': (2, operator.lt),
        '>=': (2, operator.ge),
        '<=': (2, operator.le),

        # 逻辑运算符
        'not': (6, operator.not_),
        'and': (1, operator.and_),
        'or': (0, operator.or_)
    }

    _FUNC_REGEX = re.compile(r'(?P<FUNC>float|int|str|Version)\((?P<FUNC_ARG>[/\w_.$}{\'-]*?)\)')

    _TOKEN_REGEX = re.compile(
        r'(?P<NUMBER>-?\d+(\.\d*)?)'
        r'|(?P<OP>==|!=|>=|<=|\*\*|//|>|<|\b(?:and|or|not)\b|[+\-*/%()])'
        r'|(?P<STR>(?:(\'[^\"]*?\')))'
        r'|(?P<NONE>\bNone\b)'
        r'|(?P<SKIP>\s+)'
    )
    
    _FUNC_MAP = {
        'float': float,
        'int': int,
        'str': str,
        'Version': Version
    }

    @staticmethod
    def _handle_func(mo: re.Match):
        func_name = mo.group('FUNC')
        func_arg = mo.group('FUNC_ARG')
        if isinstance(func_arg, str) and \
            func_arg.startswith("'") and \
            func_arg.endswith("'"):
            func_arg = func_arg.strip("'")
        
        repl_value = Evaluator._FUNC_MAP[func_name](func_arg)
        if isinstance(repl_value, str):
            return repr(repl_value)
        
        return f"{repl_value}"

    @staticmethod
    def _split_tokens(expr: str):
        for _ in range(5): # max func depth 5
            expr = Evaluator._FUNC_REGEX.sub(Evaluator._handle_func, expr)

        tokens = []
        has_number = False
        has_version = False

        for mo in Evaluator._TOKEN_REGEX.finditer(expr):
            kind = mo.lastgroup
            value = mo.group()

            if kind == 'SKIP':
                continue
            elif kind == 'NUMBER':
                has_number = True
                tokens.append(('NUMBER', value))
            elif kind == 'OP':
                tokens.append(('OP', value))
            elif kind == 'STR':
                tokens.append(('STR', value.strip("'")))
            elif kind == 'NONE':
                tokens.append(('NONE', value))

        # Constraint checks
        version_conflict = has_version and has_number
        if version_conflict:
            raise SyntaxError("Expression cannot contain VERSION with INT/FLOAT/NUMBER/OP tokens.")

        return tokens

    @classmethod
    def evaluate(cls, expr: str):
        if isinstance(expr, list):
            return [cls.evaluate(ex) for ex in expr]
        elif isinstance(expr, dict):
            return {k: cls.evaluate(ex) for k, ex in expr.items()}
        elif isinstance(expr, str):
            return cls._evaluate(expr)
        else:
            return expr

    @classmethod
    def _handle_tokens(cls, tokens, output, stack):
        for token in tokens:
            typ, val = token
            if typ == "NUMBER":
                output.append(float(val) if '.' in val else int(val))
            elif typ == 'OP':
                if val == '(':
                    stack.append(val)
                elif val == ')':
                    while stack and stack[-1] != '(':
                        output.append(stack.pop())
                    if not stack:
                        raise ValueError("Mismatched parentheses in expression")
                    stack.pop()
                elif val == "not":
                    stack.append(val)
                else:
                    op = val
                    prec = cls.OPS[op][0]
                    while stack and stack[-1] in cls.OPS and cls.OPS[stack[-1]][0] >= prec:
                        output.append(stack.pop())
                    stack.append(op)
            else:
                output.append(val)

    @classmethod
    def _convert_tokens_to_rpn(cls, tokens):
        # Handle expressions with INT/FLOAT and OPs
        output = []
        stack = []
        cls._handle_tokens(tokens, output, stack)

        while stack:
            if stack[-1] in ('(', ')'):
                raise ValueError("Mismatched parentheses in expression")
            output.append(stack.pop())

        return output

    @classmethod
    def _evaluate_rpn(cls, rpn):
        if len(rpn) == 1:
            return rpn[0]
        stack = []
        for token in rpn:
            if isinstance(token, (int, float, Version)):
                stack.append(token)
            elif token == "not":
                a = stack.pop()
                stack.append(cls.OPS[token][1](a))
            elif token in cls.OPS:
                b = stack.pop()
                a = stack.pop()
                stack.append(cls.OPS[token][1](a, b))
            elif isinstance(token, (str)):
                stack.append(token)
            else:
                raise ValueError(f"Unknown token in RPN: {token}")
        if len(stack) != 1:
            raise ValueError("Invalid RPN evaluation")
        return stack[0]

    @classmethod
    def _evaluate(cls, expr: Any):
        tokens = cls._split_tokens(expr)
        # in case we have a path
        if not tokens or all(t[0] == 'OP' and t[1] == r"/" for t in tokens):
            return expr

        rpn = cls._convert_tokens_to_rpn(tokens)
        result = cls._evaluate_rpn(rpn)

        return result
