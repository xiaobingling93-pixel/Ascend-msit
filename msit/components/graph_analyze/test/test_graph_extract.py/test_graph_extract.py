import os
import sys
import copy
import unittest
from argparse import Namespace
from collections import deque, Counter
from unittest.mock import MagicMock, patch, call

from onnx import ModelProto, GraphProto

from components.utils.log import logger

# Mocking the required modules and classes
sys.modules['onnx'] = MagicMock()
sys.modules['google.protobuf.text_format'] = MagicMock()
sys.modules['components.utils.log'] = MagicMock()
sys.modules['components.utils.file_open_check'] = MagicMock()

from msit_graph.graph_extract.graph_extract import GraphAnalyze, GraphSummary

class TestGraphAnalyze(unittest.TestCase):

    def setUp(self):
        # Mocking the logger and other necessary parts
        self.logger_mock = MagicMock()
        global logger
        logger = self.logger_mock

        # Mocking the onnx and text_format modules
        self.onnx_mock = sys.modules['onnx']
        self.text_format_mock = sys.modules['google.protobuf.text_format']

        # Mocking the ms_open function
        self.ms_open_mock = MagicMock()
        sys.modules['components.utils.file_open_check'].ms_open = self.ms_open_mock

        # Setting up constants
        GraphAnalyze.LOOKUP_FORWARD = 1
        GraphAnalyze.LOOKUP_BACKWARD = 2
        GraphAnalyze.LOOKUP_ALL = 3

    def test__append_file_name_suffix_given_path_without_extension_when_valid_then_correct_suffix_added(self):
        result = GraphAnalyze._append_file_name_suffix("path/to/file", "suffix")
        self.assertEqual(result, "path/to/file_suffix")

    def test__append_file_name_suffix_given_path_with_extension_when_valid_then_correct_suffix_added(self):
        result = GraphAnalyze._append_file_name_suffix("path/to/file.txt", "suffix")
        self.assertEqual(result, "path/to/file_suffix.txt")

    def test__get_node_name_given_tensor_name_with_carrot_when_valid_then_correct_name_returned(self):
        result = GraphAnalyze._get_node_name("^node_name")
        self.assertEqual(result, "node_name")

    def test__get_node_name_given_tensor_name_with_colon_when_valid_then_correct_name_returned(self):
        result = GraphAnalyze._get_node_name("node_name:1")
        self.assertEqual(result, "node_name")

    def test__get_node_name_given_tensor_name_without_special_chars_when_valid_then_correct_name_returned(self):
        result = GraphAnalyze._get_node_name("node_name")
        self.assertEqual(result, "node_name")

    def test__get_node_index_and_name_given_tensor_name_with_carrot_when_valid_then_correct_tuple_returned(self):
        result = GraphAnalyze._get_node_index_and_name("^node_name")
        self.assertEqual(result, (0, "node_name"))

    def test__get_node_index_and_name_given_tensor_name_with_colon_when_valid_then_correct_tuple_returned(self):
        result = GraphAnalyze._get_node_index_and_name("node_name:1")
        self.assertEqual(result, (1, "node_name"))

    def test__get_node_index_and_name_given_tensor_name_without_colon_when_invalid_then_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            GraphAnalyze._get_node_index_and_name("node_name")
        self.assertIn("Invalid tensor name format: node_name", str(context.exception))

    def test__get_node_index_and_name_given_tensor_name_with_non_integer_index_when_invalid_then_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            GraphAnalyze._get_node_index_and_name("node_name:abc")
        self.assertIn("Index part of tensor name node_name:abc is not an integer", str(context.exception))

    def test__lookup_dump_nodes_given_forward_lookup_when_valid_then_correct_nodes_returned(self):
        args = Namespace(stop_name=None, stop_type=None, layer_number=10, stop_leaves_count=0)
        gs = GraphSummary()
        start_node_names = ["node1"]
        lookup_directions = GraphAnalyze.LOOKUP_FORWARD

        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=[], op_type="ge:Op2")

        gs.names_to_node = {"node1": node1, "node2": node2}
        gs.names_to_output_names = {"node1": ["node2"]}
        gs.names_to_input_names = {"node2": ["node1"]}

        result = GraphAnalyze._lookup_dump_nodes(args, gs, start_node_names, lookup_directions)
        self.assertEqual(result, {"node1", "node2"})

    def test__lookup_dump_nodes_given_backward_lookup_when_valid_then_correct_nodes_returned(self):
        args = Namespace(stop_name=None, stop_type=None, layer_number=10, stop_leaves_count=0)
        gs = GraphSummary()
        start_node_names = ["node2"]
        lookup_directions = GraphAnalyze.LOOKUP_BACKWARD

        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=[], op_type="ge:Op2")

        gs.names_to_node = {"node1": node1, "node2": node2}
        gs.names_to_output_names = {"node1": ["node2"]}
        gs.names_to_input_names = {"node2": ["node1"]}

        result = GraphAnalyze._lookup_dump_nodes(args, gs, start_node_names, lookup_directions)
        self.assertEqual(result, {"node1", "node2"})

    def test__lookup_dump_nodes_given_stop_name_when_valid_then_stops_at_correct_node(self):
        args = Namespace(stop_name=["node2"], stop_type=None, layer_number=10, stop_leaves_count=0)
        gs = GraphSummary()
        start_node_names = ["node1"]
        lookup_directions = GraphAnalyze.LOOKUP_FORWARD

        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=[], op_type="ge:Op2")

        gs.names_to_node = {"node1": node1, "node2": node2}
        gs.names_to_output_names = {"node1": ["node2"]}
        gs.names_to_input_names = {"node2": ["node1"]}

        result = GraphAnalyze._lookup_dump_nodes(args, gs, start_node_names, lookup_directions)
        self.assertEqual(result, {"node1"})

    def test__lookup_dump_nodes_given_stop_type_when_valid_then_stops_at_correct_op_type(self):
        args = Namespace(stop_name=None, stop_type=["ge:Op2"], layer_number=10, stop_leaves_count=0)
        gs = GraphSummary()
        start_node_names = ["node1"]
        lookup_directions = GraphAnalyze.LOOKUP_FORWARD

        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=[], op_type="ge:Op2")

        gs.names_to_node = {"node1": node1, "node2": node2}
        gs.names_to_output_names = {"node1": ["node2"]}
        gs.names_to_input_names = {"node2": ["node1"]}

        result = GraphAnalyze._lookup_dump_nodes(args, gs, start_node_names, lookup_directions)
        self.assertEqual(result, {"node1"})

    def test__generate_graph_given_backbone_names_when_valid_then_generates_correct_subgraph(self):
        gs = GraphSummary()
        backbone_names = ["node1", "node2"]
        output_path = "path/to/output.pbtxt"
        without_leaves = False
    
        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=[], op_type="ge:Op2")
    
        node1.name = "node1"
        node2.name = "node2"
    
        gs.names_to_node = {"node1": node1, "node2": node2}
        gs.names_to_output_names = {"node1": ["node2"]}
        gs.names_to_input_names = {"node2": ["node1"]}
        gs.names_to_seq_num = {"node1": 0, "node2": 1}
    
        GraphAnalyze._save_graph_def = MagicMock()
    
        GraphAnalyze._generate_graph(gs, backbone_names, output_path, without_leaves)
    
        GraphAnalyze._save_graph_def.assert_called_once()

    #def test__find_nodes_between_start_and_end_given_valid_start_and_end_names_when_valid_then_returns_correct_nodes(self)

    #def test__find_nodes_by_prefixes_given_valid_prefixes_when_valid_then_returns_correct_nodes

    def test_extract_sub_graph_given_valid_args_when_valid_then_generates_correct_subgraph(self):
        args = Namespace(input="path/to/input.pbtxt", output=None, start_node=["node1"], end_node=["node3"],
                         name_prefix=["node"], without_leaves=False)
        gs = GraphSummary()
        input_path = "path/to/input.pbtxt"
        output_path = "path/to/input_sub.pbtxt"

        node1 = MagicMock(name="node1", input=[], output=["node2"], op_type="ge:Op1")
        node2 = MagicMock(name="node2", input=["node1"], output=["node3"], op_type="ge:Op2")
        node3 = MagicMock(name="node3", input=["node2"], output=[], op_type="ge:Op3")

        gs.names_to_node = {"node1": node1, "node2": node2, "node3": node3}
        gs.names_to_output_names = {"node1": ["node2"], "node2": ["node3"]}
        gs.names_to_input_names = {"node2": ["node1"], "node3": ["node2"]}

        graph_def_mock = MagicMock(node=[node1, node2, node3])
        GraphAnalyze.load_graph_def_from_pbtxt = MagicMock(return_value=graph_def_mock)
        GraphAnalyze._build_graph_summary = MagicMock(return_value=gs)
        GraphAnalyze._find_nodes_by_start_names = MagicMock(return_value={"node1", "node2"})
        GraphAnalyze._find_nodes_between_start_and_end = MagicMock(return_value={"node2", "node3"})
        GraphAnalyze._find_nodes_by_prefixes = MagicMock(return_value={"node1", "node2"})
        GraphAnalyze._generate_graph = MagicMock()

        result = GraphAnalyze.extract_sub_graph(args)

        self.assertEqual(result, 0)
        GraphAnalyze._generate_graph.assert_called_once_with(gs, {"node1", "node2", "node3"}, output_path, False)

