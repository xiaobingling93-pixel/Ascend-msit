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

import re
from functools import partial


class ExpandError(Exception):
    def __init__(self, expr):
        self.expr = expr
        super().__init__(f"Cannot expand {expr}")


class MacroExpander:
    DOT = "."
    VAR_REGEX = re.compile(r"\$\{([\w_.\[\]/]+?)\}")

    @classmethod
    def expand(cls, expr, context, visited_path):
        if isinstance(expr, dict):
            return {k: cls.expand(v, context, visited_path) for k, v in expr.items()}
        elif isinstance(expr, list):
            return [cls.expand(item, context, visited_path) for item in expr]
        elif isinstance(expr, str):
            full_path = cls.VAR_REGEX.sub(partial(cls._expand_var, context=context, visited_path=visited_path), expr)
            return full_path
        else:
            return expr

    @classmethod
    def _expand_var(cls, match_object: re.Match, context, visited_path):
        path = match_object.group(1)
        if not path:
            raise ExpandError(path)

        full_path = cls._build_full_path(path, context)
        if full_path not in visited_path:
            raise ExpandError(full_path)

        val = visited_path[full_path]
        if isinstance(val, str):
            return repr(val)

        return str(val)

    @classmethod
    def _build_full_path(cls, path: str, context: str) -> str:
        if path == cls.DOT:
            return context

        if path.startswith(cls.DOT):
            full_path = context + path
            comps = full_path.split('.')
        
            stack = []
            for comp in comps:
                if comp:
                    stack.append(comp)
                elif not stack:
                    raise ExpandError(full_path)
                else:
                    stack.pop()

            return '.'.join(stack)

        return path
