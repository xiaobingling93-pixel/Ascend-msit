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

from collections import deque


class Traverser:
    @staticmethod
    def traverse(data):
        if not isinstance(data, (dict, list)):
            raise TypeError(f"Expected 'data' to be dict or list. Got {type(data).__name__} instead.")
            
        queue = deque()
        queue.append((data, ""))

        visited_nodes = {}

        while queue:
            node, path = queue.popleft()

            if path:
                visited_nodes[path] = node

            if isinstance(node, dict):
                Traverser._traverse_dict(node, path, queue)
            elif isinstance(node, list):
                Traverser._traverse_list(node, path, queue)

        return visited_nodes

    @staticmethod
    def _traverse_dict(node: dict, path: str, queue: deque):
        for k, v in node.items():
            new_path = k if not path else path + "." + k
            queue.append((v, new_path))

    @staticmethod
    def _traverse_list(node: list, path: str, queue: deque):
        for i, v in enumerate(node):
            new_path = f"[{i}]" if not path else path + f"[{i}]"
            queue.append((v, new_path))
