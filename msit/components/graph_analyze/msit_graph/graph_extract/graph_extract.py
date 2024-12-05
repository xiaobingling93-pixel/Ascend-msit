# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import argparse

from collections import Counter, defaultdict, deque

import onnx
from google.protobuf import text_format

from components.utils.log import logger
from components.utils.file_open_check import ms_open, OpenException, 
    MAX_SIZE_LIMITE_CONFIG_FILE, MAX_SIZE_LIMITE_NORMAL_FILE


class GraphSummary:
    def __init__(self):
        # Maps node names to their input names
        self.names_to_input_names = {}
        # Maps node names to their data input names
        self.names_to_data_input_names = {}
        # Maps node names to their control input names
        self.names_to_ctrl_input_names = {}
        # Maps node names to their output names (list of outputs)
        self.names_to_output_names = defaultdict(list)
        # Maps node names to their corresponding ONNX Node objects
        self.names_to_node = {}
        # Keeps track of node sequences for maintaining original order
        self.names_to_seq_num = {}


class GraphAnalyze:

    LOOKUP_FORWARD = 1
    LOOKUP_BACKWARD = 2
    LOOKUP_ALL = 3

    def __init__(self):
        pass

    @staticmethod
    def _append_file_name_suffix(path, suffix):
        """Appends a suffix to the base name of the file in the given path."""
        tokens = path.rsplit('.', 1)  # Split the path by the last occurrence of '.'
        if len(tokens) == 1:
            # If there is no '.', treat the whole path as the base name
            return tokens[0] + '_' + suffix
        else:
            # If there is an extension, split into base name and extension
            base_name, extension = tokens
            return f"{base_name}_{suffix}.{extension}"

    @staticmethod
    def _load_graph_def_from_pbtxt(path):
        """Loads an ONNX graph definition from a binary protocol buffer file."""
        logger.info(f"Loading {path}, the graph maybe huge, please wait a minute...")
        try:
            with ms_open(path, "rb", MAX_SIZE_LIMITE_NORMAL_FILE) as f:
                data = f.read()
                model = onnx.ModelProto()
                text_format.Parse(data, model)
                logger.info(f"Load {path} success!")
                return model.graph
        except OpenException as oe:
            logger.error(f"OpenException occurred: {oe}")
        except FileNotFoundError as fnf:
            logger.error(f"File not found: {fnf}")
        except PermissionError as pe:
            logger.error(f"Permission error: {pe}")
        except text_format.ParseError as pe:
            logger.error(f"Parse error: {pe}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return None

    @staticmethod
    def _save_graph_def(graph_def, path, as_text=False):
        """Saves an ONNX graph definition to a file, optionally in text format."""
        model_def = onnx.helper.make_model(graph_def, producer_name='onnx-subgraph')
        if as_text:
            try:
                with ms_open(path, "w") as f:
                    f.write(text_format.MessageToString(model_def))
            except OpenException as oe:
                logger.error(f"OpenException occurred: {oe}")
            except FileNotFoundError as fnf:
                logger.error(f"File not found: {fnf}")
            except PermissionError as pe:
                logger.error(f"Permission error: {pe}")
            except text_format.ParseError as pe:
                logger.error(f"Parse error: {pe}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
        else:
            onnx.save(model_def, path)

    @staticmethod
    def _get_node_name(tensor_name):
        """Extracts the node name from a tensor name."""
        if tensor_name.startswith("^"):
            return tensor_name[1:]
        return tensor_name.split(":")[0]

    @staticmethod
    def _get_node_index_and_name(tensor_name):
        """Extracts the index and name from a tensor name."""
        if tensor_name.startswith("^"):
            return 0, tensor_name[1:]
        if not tensor_name:
            return 0, ""
        name_and_index = tensor_name.rsplit(":", 1)
        if len(name_and_index) != 2:
            logger.warning(f"Invalid node input {tensor_name}")
            raise ValueError(f"Invalid tensor name format: {tensor_name}")
        
        name, index = name_and_index
        try:
            return int(index), name
        except ValueError:
            logger.error(f"Index part of tensor name {tensor_name} is not an integer")
            raise ValueError(f"Index part of tensor name {tensor_name} is not an integer") from e

    @staticmethod
    def _nodes_to_gs(nodes, gs, start_seq):
        """
        Populates a GraphSummary object with information about nodes.
        
        Args:
            nodes (list): List of nodes to process.
            gs (GraphSummary): The GraphSummary object to populate.
            start_seq (int): Starting sequence number for node indexing.
            
        Returns:
            int: The next sequence number after processing all nodes.
        """
        seq = start_seq
        for node in nodes:
            node_name = GraphAnalyze._get_node_name(node.name)
            gs.names_to_node[node_name] = node
            input_indexes_and_names = [GraphAnalyze._get_node_index_and_name(input_name) for input_name in node.input]
            # Separate data inputs and control inputs based on index
            gs.names_to_data_input_names[node_name] = [
                index_and_name[1] for index_and_name in input_indexes_and_names if index_and_name[0] >= 0
            ]
            gs.names_to_ctrl_input_names[node_name] = [
                index_and_name[1] for index_and_name in input_indexes_and_names if index_and_name[0] < 0
            ]
            gs.names_to_ctrl_input_names[node_name].sort()
            gs.names_to_input_names[node_name] = (
                gs.names_to_data_input_names[node_name] + gs.names_to_ctrl_input_names[node_name]
            )

            if not node.output and gs.names_to_output_names.get(node_name):
                logger.warning(f"The node {node_name} does not have output record, but found input record from others")

            for input_name in node.input:
                gs.names_to_output_names[GraphAnalyze._get_node_name(input_name)].append(node_name)
            gs.names_to_seq_num[node_name] = seq
            seq += 1

            for attr in node.attribute:
                if attr.name == "graph":
                    seq = GraphAnalyze._nodes_to_gs(attr.g.node, gs, seq)

        return seq

    @staticmethod
    def _build_graph_summary(graph_def):
        """Builds a GraphSummary object from a graph definition."""
        gs = GraphSummary()
        seq = 0
        GraphAnalyze._nodes_to_gs(graph_def.node, gs, 0)
        return gs

    @staticmethod
    def _lookup_dump_nodes(args, gs, start_node_names, lookup_directions):
        """
        Finds nodes to dump based on specified criteria.
        
        Args:
            args (Namespace): Command line arguments containing stop_name, stop_type, layer_number, stop_leaves_count.
            gs (GraphSummary): The GraphSummary object containing graph information.
            start_node_names (list): List of starting node names for lookup.
            lookup_directions (int): Bitmask indicating lookup direction (forward or backward).
            
        Returns:
            set: Set of node names that match the criteria.
        """
        dump_tasks = deque()
        stop_name = args.stop_name
        stop_type = args.stop_type

        # Initialize stop names and types sets
        if not stop_name:
            stop_names = set()
        else:
            stop_names = set(stop_name)

        if not stop_type:
            stop_types = set()
        else:
            stop_types = set(stop_type)

        # Add initial tasks based on lookup directions
        for start_node_name in start_node_names:
            if lookup_directions & GraphAnalyze.LOOKUP_FORWARD:
                dump_tasks.append((start_node_name, 'forward', 0))
            if lookup_directions & GraphAnalyze.LOOKUP_BACKWARD:
                dump_tasks.append((start_node_name, 'backward', 0))

        dump_node_names = set()

        # Process tasks until none remain
        while dump_tasks:
            node_name, direction, depth = dump_tasks.popleft()
            
            if not node_name or node_name in stop_names:
                continue
            
            node = gs.names_to_node.get(node_name)
            if not node or node.op_type in stop_types:
                continue

            if depth >= args.layer_number:
                continue

            dump_node_names.add(node_name)
            stop_leaves_count = args.stop_leaves_count

            # Extend tasks based on the direction of lookup
            if direction == 'forward':
                output_node_names = gs.names_to_output_names.get(node_name, [])
                if stop_leaves_count == 0 or len(output_node_names) < stop_leaves_count:
                    dump_tasks.extend(
                        (output_node_name, direction, depth + 1) for output_node_name in output_node_names
                    )
            else:
                input_node_names = gs.names_to_input_names.get(node_name, [])
                if stop_leaves_count == 0 or len(input_node_names) < stop_leaves_count:
                    dump_tasks.extend(
                        (input_node_name, direction, depth + 1) for input_node_name in input_node_names
                    )

        return dump_node_names

    @staticmethod
    def _generate_graph(gs, backbone_names, output_path, without_leaves):
        """
        Generates a sub-graph based on specified backbone nodes.
        
        Args:
            gs (GraphSummary): The GraphSummary object containing graph information.
            backbone_names (list): List of backbone node names to include in the sub-graph.
            output_path (str): Path where the generated sub-graph will be saved.
            without_leaves (bool): If True, do not include leaf nodes in the sub-graph.
            
        Returns:
            None
        """
        dump_names = set()
        for backbone in backbone_names:
            if not backbone:
                continue
            dump_names.add(backbone)
            if not without_leaves:
                # Add input and output nodes of the backbone nodes if including leaves
                for input_name in gs.names_to_input_names.get(backbone, []):
                    dump_names.add(input_name)
                for output_name in gs.names_to_output_names.get(backbone, []):
                    dump_names.add(output_name)

        dump_nodes = []
        for dump_name in dump_names:
            dump_node = gs.names_to_node.get(dump_name, None)
            if dump_node:
                dump_nodes.append(dump_node)
        dump_nodes.sort(key=lambda node: gs.names_to_seq_num[node.name])

        out = onnx.GraphProto()
        for dump_node in dump_nodes:
            out.node.extend([copy.deepcopy(dump_node)])

        logger.info(f"Save to {output_path}")
        logger.info(f"Total nodes = {len(out.node)}")
        GraphAnalyze._save_graph_def(out, output_path, as_text=True)

    @staticmethod
    def _find_nodes_by_start_names(args, gs):
        """Finds nodes starting from specified names."""
        if not args.name:
            return set()
        lookup_directions = GraphAnalyze.LOOKUP_ALL
        if args.only_forward:
            lookup_directions ^= GraphAnalyze.LOOKUP_BACKWARD
        if args.only_backward:
            lookup_directions ^= GraphAnalyze.LOOKUP_FORWARD
        if not lookup_directions:
            logger.error("The --only_forward and --only_backward cannot exist at the same time")
            return set()

        start_nodes = [node_name for node_name in args.name if node_name in gs.names_to_node]

        logger.info(f"Begin to find dump nodes, start nodes {start_nodes}")
        return GraphAnalyze._lookup_dump_nodes(
            args, gs, start_nodes, lookup_directions
        )

    @staticmethod
    def _find_nodes_between_start_and_end(gs, start_names=None, end_names=None):
        """
        Finds nodes between specified start and end names.
        
        Args:
            gs (GraphSummary): The GraphSummary object containing graph information.
            start_names (list): List of starting node names.
            end_names (list): List of ending node names.
            
        Returns:
            set: Set of node names that are between the specified start and end nodes.
        """
        if start_names is None or end_names is None:
            logger.error("Both --start_name and --end_name must be provided")
            return set()

        if len(start_names) != len(end_names):
            logger.error("The number of --start_name and the --end_name must be the same")
            return set()

        logger.info("Begin to lookup nodes by start and end nodes...")
        dump_nodes = set()

        for start, end in zip(start_names, end_names):
            nodes_to_end = set()
            que = deque([end])
            
            while que:
                node_name = que.popleft()
                if not node_name or node_name in nodes_to_end or node_name == start:
                    continue
                nodes_to_end.add(node_name)
                que.extend(gs.names_to_input_names.get(node_name, []))

            dump_nodes_this_pattern = set()
            que = deque([start])
            
            while que:
                node_name = que.popleft()
                if node_name in dump_nodes_this_pattern or node_name == end or node_name not in nodes_to_end:
                    continue
                dump_nodes_this_pattern.add(node_name)
                que.extend(gs.names_to_output_names.get(node_name, []))

            dump_nodes_this_pattern.add(end)
            dump_nodes.update(dump_nodes_this_pattern)

        return dump_nodes

    @staticmethod
    def _find_nodes_by_prefixes(gs, name_prefix):
        """Finds nodes matching specified prefixes."""
        if name_prefix is None:
            return set()
        logger.info("Begin to lookup nodes by name prefixes...")
        dump_nodes = set()
        for name in gs.names_to_node.keys():
            if any(name.startswith(prefix) for prefix in name_prefix):
                dump_nodes.add(name)
                break
        return dump_nodes

    @staticmethod
    def extract_sub_graph(args):
        """Extracts a sub-graph from the input graph based on specified criteria."""
        input_path = args.input
        output_path = args.output
        if not output_path:
            output_path = GraphAnalyze._append_file_name_suffix(input_path, "sub")

        logger.info(f"Begin to read in graph from file {input_path}")
        graph_def = GraphAnalyze._load_graph_def_from_pbtxt(input_path)

        gs = GraphAnalyze._build_graph_summary(graph_def)

        dump_node_names = GraphAnalyze._find_nodes_by_start_names(args, gs)
        dump_node_names.update(GraphAnalyze._find_nodes_between_start_and_end(gs, args.start_node, args.end_node))
        dump_node_names.update(GraphAnalyze._find_nodes_by_prefixes(gs, args.name_prefix))

        if not dump_node_names:
            logger.error("No nodes to dump")
            return -1
        GraphAnalyze._generate_graph(gs, dump_node_names, output_path, args.without_leaves)
        return 0

    @staticmethod
    def find_nodes_by_type(input_path, node_type=None):
        """Finds nodes in the graph by their operation type."""
        logger.info(f"Begin to read in graph from file {input_path}")
        graph_def = GraphAnalyze._load_graph_def_from_pbtxt(input_path)

        node_types_to_names = {}
        if node_type is not None:
            node_type = set(node_type)
            for node in graph_def.node:
                if node.op_type in node_type:
                    node_types_to_names.setdefault(node.op_type, []).append(node.name)
        for node_type, node_names in node_types_to_names.items():
            logger.info(f"Node type {node_type}:")
            node_names.sort()
            for node_name in node_names:
                logger.info(f"  {node_name}")

    @staticmethod
    def strip(input_path, output_path=None):
        """Strips Cons/ Data Node and Attribute from an ONNX model file."""
        if not output_path:
            output_path = GraphAnalyze._append_file_name_suffix(input_path, "sub")
        logger.info(f"Begin to read from {input_path} and write to {output_path}")
        
        try:
            line_count, drop_count, c_count, drop_c_count = GraphAnalyze._process_file(input_path, output_path)
        except OpenException as oe:
            logger.error(f"OpenException occurred: {oe}")
        except FileNotFoundError as fnf:
            logger.error(f"File not found: {fnf}")
        except PermissionError as pe:
            logger.error(f"Permission error: {pe}")
        except text_format.ParseError as pe:
            logger.error(f"Parse error: {pe}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        finally:
            logger.info(f"Dropped lines count {drop_count}, char size {drop_c_count}; total lines count {line_count}, char size {c_count}")

    @staticmethod
    def _process_file(input_path, output_path):
        """
        Processes a subgraph by iterating through its nodes, copying nodes that are not of type
        "ge:Const" or "ge:Data", and removes all attributes from the copied nodes.

        Args:
            subgraph (onnx.GraphProto): The subgraph to process.
        """
        graph = _load_graph_def_from_pbtxt(input_path)
        logging.info(f"\t total node = {len(graph.node)}")

        out = onnx.GraphProto()
        nodes_copy = []

        def process_node(node):
            if node.op_type not in ("ge:Const", "ge:Data"):
                node_copy = copy.deepcopy(node)
                node_copy.ClearField('attribute')
                nodes_copy.append(node_copy)

        def process_subgraph(subgraph):
            subgraph_nodes = []
            for subnode in subgraph.node:
                if subnode.op_type not in ("ge:Const", "ge:Data"):
                    subnode_copy = copy.deepcopy(subnode)
                    subnode_copy.ClearField('attribute')  # Remove all attributes
                    subgraph_nodes.append(subnode_copy)
            subgraph.ClearField('node')
            subgraph.node.extend(subgraph_nodes)

        for node in graph.node:
            if node.op_type == "subgraph":
                node_copy = copy.deepcopy(node)
                for attr in node_copy.attribute:
                    if attr.HasField('g'):  # Check if the attribute has a subgraph
                        process_subgraph(attr.g)
                nodes_copy.append(node_copy)
            else:
                process_node(node)

        nodes_copy.sort(key=lambda node: node.name)
        out.node.extend(nodes_copy)
        logging.info(f"save to {output_path}")
        logging.info(f"total node = {len(out.node)}")

        # Save graph
        model_def = onnx.helper.make_model(out, producer_name='onnx-subgraph')
        with open(output_path, 'w') as f:
            f.write(text_format.MessageToString(model_def))

    @staticmethod
    def print_graph_stat(input_path):
        """Prints statistics about the operations in an ONNX graph."""
        graph_def = GraphAnalyze._load_graph_def_from_pbtxt(input_path)

        op_stat = Counter()
        
        def process_node(node):
            op_stat[node.op_type] += 1
            for attr in node.attribute:
                if attr.HasField('g'):  # Check if the attribute has a subgraph
                    for subnode in attr.g.node:
                        process_node(subnode)
        
        for node in graph_def.node:
            process_node(node)

        logger.info("Graph stats:")
        for op, count in sorted(op_stat.items(), key=lambda x: x[0]):
            logger.info(f"\t{op} = {count}")
