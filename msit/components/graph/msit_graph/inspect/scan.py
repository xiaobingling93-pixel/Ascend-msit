# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import os

from components.utils.log import logger
from components.utils.file_open_check import ms_open
from components.utils.constants import CSV_FILE_MAX_SIZE
from msit_graph.graph_extract.graph_extract import GraphAnalyze
from components.utils.constants import MAX_DEPTH_LIMIT


def save_dym_op(data, path):
    with ms_open(path, 'w', CSV_FILE_MAX_SIZE) as f:
        f.write(",".join(["Graph_Name", "Node_Name", "Input", "Output"]) + "\n")
        for row in data:
            f.write(",".join(map(str, row)) + "\n")
    logger.info("The list of dynamic shape operator saved in %r." % path)


class DynamicShape:
    def __init__(self, graph):
        self.graph = graph
        self.dynamic_to_static_edges = []
        self.graph_name = []

    @staticmethod
    def is_dynamic_shape(node):
        def is_dynamic_shape_attr(attr):
            return (attr.name == "_is_unknown_shape" or attr.name == "_force_unknown_shape") and attr.i == 1
        for attr in node.attribute:
            if is_dynamic_shape_attr(attr):
                return True
        return False

    def add_dynamic_op(self, parent_name, sub_node_name, inputs, outputs):
        dynamic_op = (parent_name, sub_node_name, inputs, outputs)
        if dynamic_op not in self.dynamic_to_static_edges and parent_name not in self.graph_name:
            self.dynamic_to_static_edges.append(dynamic_op)
            self.graph_name.append(parent_name)

    def process_node(self, node, parent_name=None, depth=0):
        if depth > MAX_DEPTH_LIMIT:
            raise RecursionError(
                f"Exceeded maximum recursion depth {MAX_DEPTH_LIMIT} when process node"
            )
        if self.is_dynamic_shape(node) and parent_name:
            node.input[:] = [";".join(node.input)]
            node.output[:] = [";".join(node.output)]
            self.add_dynamic_op(parent_name, node.name, node.input, node.output)
        for attr in node.attribute:
            if hasattr(attr, 'g') and attr.g:
                for sub_node in attr.g.node:
                    self.process_node(sub_node, node.name, depth=depth + 1)

    def find_dynamic_shape_op(self):
        for node in self.graph.node:
            self.process_node(node)
        return self.dynamic_to_static_edges


def execute(args):
    if args.type == "dshape":
        pb_graph = GraphAnalyze.load_graph_def_from_pbtxt(args.input)
        dym_ops = DynamicShape(pb_graph).find_dynamic_shape_op()
        if os.path.exists(args.output) and os.path.isdir(args.output):
            save_dym_op(dym_ops, os.path.join(args.output, "dynamic_shape_ops.csv"))
        else:
            raise Exception("Please check if the directory exists.")
