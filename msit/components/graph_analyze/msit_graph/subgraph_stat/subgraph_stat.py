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

from collections import Counter
from collections import defaultdict, deque
import os

from itertools import combinations
import pandas as pd

from components.utils.log import logger
from components.utils.file_open_check import (
    ms_open, 
    OpenException, 
    MAX_SIZE_LIMITE_CONFIG_FILE,
    MAX_SIZE_LIMITE_NORMAL_FILE,
)
from msit_graph.graph_extract.graph_extract import GraphAnalyze


class SimpleNode:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.inputs = []
        self.outputs = []


def parse_pbtxt(graph_def):
    nodes = {}
    """
    Parse the graph definition from a .pbtxt file into a dictionary of SimpleNode objects.

    :param graph_def: The parsed GraphDef object from the .pbtxt file.
    :return: A dictionary where keys are node names and values are SimpleNode objects.
    """
    # First pass: Create all nodes and collect their inputs
    for node in graph_def.node:
        node_name = node.name
        node_type = node.op_type
        nodes[node_name] = SimpleNode(node_name, node_type)

        inputs = [input_name.split(':')[0] for input_name in node.input]
        nodes[node_name].inputs = inputs

    # Second pass: Populate outputs based on inputs
    for node_name, node in nodes.items():
        for input_node_name in node.inputs:
            if input_node_name in nodes:
                nodes[input_node_name].outputs.append(node_name)

    return nodes


def bfs_subgraph(root_name, nodes, max_nodes=10):
    """
    Perform a breadth-first search (BFS) starting from root_name to find a subgraph
    with up to max_nodes nodes.

    :param root_name: The name of the root node to start BFS from.
    :param nodes: Dictionary of SimpleNode objects representing the graph.
    :param max_nodes: Maximum number of nodes in the resulting subgraph.
    :return: List of node names forming the subgraph.
    """
    queue = deque([root_name])
    visited = set()
    subgraph_nodes = []

    while queue and len(subgraph_nodes) < max_nodes:
        current_name = queue.popleft()
        if current_name not in visited:
            visited.add(current_name)
            subgraph_nodes.append(current_name)
            for neighbor_name in nodes[current_name].outputs:
                if neighbor_name not in visited:
                    queue.append(neighbor_name)

    return subgraph_nodes


def get_subtree(nodes, node_name):
    """
    Get all nodes in the subtree rooted at node_name.

    :param nodes: Dictionary of SimpleNode objects representing the graph.
    :param node_name: The name of the root node of the subtree.
    :return: Set of node names in the subtree.
    """
    subtree = set()
    stack = [node_name]
    while stack:
        current = stack.pop()
        if current not in subtree:
            subtree.add(current)
            for output in nodes[current].outputs:
                stack.append(output)
    return subtree


def generate_subgraphs(root_name, nodes, bfs_nodes):
    """
    Generate all possible subgraphs by removing non-root nodes iteratively.

    :param root_name: The name of the root node.
    :param nodes: Dictionary of SimpleNode objects representing the graph.
    :param bfs_nodes: List of node names forming the initial BFS-generated subgraph.
    :return: Set of tuples representing unique subgraph paths.
    """
    subgraphs = set()
    non_root_nodes = set(bfs_nodes) - {root_name}

    # Add the initial BFS subgraph as the first subgraph
    initial_subgraph_types = tuple(nodes[node_name].type for node_name in bfs_nodes)
    subgraphs.add(initial_subgraph_types)

    # Generate all subsets of non-root nodes
    for r in range(1, len(non_root_nodes) + 1):
        for subset in combinations(non_root_nodes, r):
            removed_nodes = set(subset)
            # Remove the subtree rooted at each node in the subset
            for node in subset:
                removed_nodes.update(get_subtree(nodes, node))

            # Calculate the remaining nodes in the subgraph while maintaining order
            remaining_nodes = list(filter(lambda node: node not in removed_nodes, bfs_nodes))

            # Calculate the remaining nodes in the subgraph
            remaining_nodes = set(bfs_nodes) - removed_nodes
            # Ensure the subgraph contains at least two nodes: root and one child
            if len(remaining_nodes) >= 2:
                subgraph_types = tuple(nodes[node_name].type for node_name in remaining_nodes)
                logger.debug(f"subgraph_types: {subgraph_types}")
                subgraphs.add(subgraph_types)

    return subgraphs


def find_duplicate_subgraphs(graphs, max_nodes=10):
    """
    Find and count all duplicate subgraphs in the given graph.

    :param nodes: Dictionary of SimpleNode objects representing the graph.
    :param subgraph_count: Dictionary to store counts of each subgraph.
    :param max_nodes: Maximum number of nodes in the subgraphs.
    """
    total_count = sum(len(nodes) for nodes in graphs)
    index_counter = 1
    subgraph_count = defaultdict(int)
    for nodes in graphs:
        for node_name in nodes.keys():
            bfs_nodes = bfs_subgraph(node_name, nodes, max_nodes)
            logger.debug(f"------bfs_nodes {bfs_nodes}")
            logger.info(f"processing node {node_name}, {index_counter}/{total_count}")
            subgraphs = generate_subgraphs(node_name, nodes, bfs_nodes)
            logger.debug(f"subgraphs:{subgraphs}")
            index_counter += 1
            for subgraph in subgraphs:
                subgraph_count[subgraph] += 1
    return subgraph_count


def has_subgraph(graph):
    for node in graph.node:
        if node.op_type == "subgraph":
            return True
    return False


def stat_subgraph(input_path, max_nodes=10, output_file='subgraph_counts.csv'):
    graph_def = GraphAnalyze.load_graph_def_from_pbtxt(input_path)
    if graph_def is None:
        logger.info(f"Failed to parse the pbtxt file.")
        return

    graphs = []
    if has_subgraph(graph_def):
        for node in graph_def.node:
            if node.op_type == "subgraph":
                for attr in node.attribute:
                    if attr.HasField('g'):  # Check if the attribute has a subgraph
                        nodes = parse_pbtxt(attr.g)
                        if nodes is not None:
                            graphs.append(nodes)
                        else:
                            logger.info(f"Failed to parse the pbtxt file.")
    else:
        nodes = parse_pbtxt(graph_def)
        if nodes is not None:
            graphs.append(nodes)
        else:
            logger.info(f"Failed to parse the pbtxt file.")

    subgraph_count = find_duplicate_subgraphs(graphs, max_nodes)
    # Sort duplicate subgraphs by their count in descending order
    duplicate_subgraphs = sorted(subgraph_count.items(), key=lambda x: x[1], reverse=True)
    # Prepare data for DataFrame
    data = {'Subgraph': [hash_value for hash_value, _ in duplicate_subgraphs],
            'Count': [count for _, count in duplicate_subgraphs]}

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save DataFrame to Excel
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")


def calculate_average_durations(input2_df):
    # Group by 'OP Type' and calculate the mean of 'Task Duration(us)'
    average_durations = input2_df.groupby('OP Type')['Task Duration(us)'].mean().reset_index()
    average_durations.columns = ['OP Type', 'Average Task Duration(us)']
    return average_durations


def preprocess_subgraph(subgraph_str):
    # Remove 'ge:' prefix from each OP Type in the subgraph tuple
    op_types = eval(subgraph_str)
    cleaned_op_types = tuple(op_type.replace('ge:', '') for op_type in op_types)
    return str(cleaned_op_types)


def calculate_sum(input1_path, input2_path, output_path='subgraph_duration_stat.csv'):
    # Read input CSV files
    input1_df = pd.read_csv(input1_path)
    input2_df = pd.read_csv(input2_path)

    # Preprocess Subgraph column in input1_df
    input1_df['Subgraph'] = input1_df['Subgraph'].apply(preprocess_subgraph)

    # Calculate average durations for each OP Type
    average_durations = calculate_average_durations(input2_df)

    # Function to sum up the task durations based on the subgraph hash
    def sum_task_durations(subgraph_hash, average_durations):
        op_types = eval(subgraph_hash)  # Convert string representation of tuple to actual tuple
        total_duration = 0.0
        for op_type in op_types:
            avg_duration = average_durations.loc[average_durations['OP Type'] == op_type, 
                'Average Task Duration(us)'].values
            if len(avg_duration) > 0:
                total_duration += avg_duration[0]
        return total_duration

    # Apply the function to create 'Task Sum Duration(us)' column
    input1_df['Task Sum Duration(us)'] = input1_df['Subgraph'].apply(lambda x: sum_task_durations(x, average_durations))

    # Calculate 'Total Duration(us)' column
    input1_df['Total Duration(us)'] = input1_df['Count'] * input1_df['Task Sum Duration(us)']

    # Select only the required columns for the output
    output_df = input1_df[['Subgraph', 'Count', 'Task Sum Duration(us)', 'Total Duration(us)']]

    # Save the result to output CSV file
    output_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")