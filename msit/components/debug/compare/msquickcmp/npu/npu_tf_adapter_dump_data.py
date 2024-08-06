#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

"""
Function:
This class mainly involves generate tf adapter npu dump data function.
"""
import os

import npu_device
import numpy as np
import tensorflow as tf
import json

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from msquickcmp.common import utils
from npu_device.compat.v1.npu_init import *


class NpuTfAdapterDumpData(object):
    """
    This class is used to generate GUP dump data of the tf2.6 save_model.
    """

    def __init__(self, arguments):
        output_path = os.path.realpath(arguments.out_path)
        self.input = os.path.join(output_path, "input")
        self.dump_data_npu = os.path.join(output_path, "dump_data", "npu")
        self.inputs_data = {}
        self.model_path = arguments.offline_model_path
        self.input_shape = self.split_input_shape(arguments.input_shape)
        self.net_output = {}
        self._create_dir()

    @staticmethod
    def _parse_json_file(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File '{output_json_path}' not found, Please check whether the json file path is "
                                    f"valid. {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"File '{output_json_path}' is not a valid JSON format. {e}") from e

    @staticmethod
    def split_input_shape(input_shapes):
        input_list = input_shapes.split(";")
        input_shape_list = [
            (name, [int(num) for num in shape_data_str_list])
            for name, shape_data_str in (shape.split(":") for shape in input_list)
            for shape_data_str_list in [shape_data_str.split(",")]
        ]

        return input_shape_list

    @staticmethod
    def _is_op_exists(operation_name_to_check, operations):
        return any(op.name == operation_name_to_check for op in operations)

    def _create_dir(self):
        utils.create_directory(self.input)
        utils.create_directory(self.dump_data_npu)

    def generate_inputs_data(self):
        for shape in self.input_shape:
            input_data = np.random.random(size=shape[1]).astype(np.float32)
            self.inputs_data[shape[0]] = input_data
            input_data.tofile(os.path.join(self.input, shape[0] + ".bin"))

    def generate_dump_data(self, output_json_path):
        graph = self._load_graph()
        # adapt NPU
        npu_device.compat.enable_v1()
        config_proto = tf.compat.v1.ConfigProto()
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        tf_config = npu_config_proto(config_proto=config_proto)

        # sess run predict
        with tf.compat.v1.Session(config=tf_config, graph=graph) as sess:
            feed_dict = {
                sess.graph.get_tensor_by_name(input_name + ":0"): input_data
                for input_name, input_data in self.inputs_data.items()
            }
            op_names = self.parse_ops_name_from_om_json(output_json_path)
            output_tensors = []
            operations = sess.graph.get_operations()
            for op_name in op_names:
                if self._is_op_exists(op_name, operations):
                    output_tensors.extend(sess.graph.get_operation_by_name(op_name).outputs)
            out = sess.run(output_tensors, feed_dict)
            self._save_dump_data(out, output_tensors)
            utils.logger.info("Dump tf adapter data success, data saved in: %s", self.dump_data_npu)

        return self.dump_data_npu

    def _load_graph(self):
        sess = tf.compat.v1.keras.backend.get_session()
        tag_set = {tf.compat.v1.saved_model.tag_constants.SERVING}
        _ = tf.compat.v1.saved_model.load(sess, tag_set, self.model_path)
        return sess.graph

    def parse_ops_name_from_om_json(self, output_json_path):
        op_names = []
        om = self._parse_json_file(output_json_path)
        graph_list = om.get('graph')
        for graph in graph_list:
            ops = graph.get('op', [])
            output_desc_list = [op.get('output_desc', []) for op in ops]
            attrs_list = [od.get('attr', []) for od in sum(output_desc_list, [])]
            for attr in sum(attrs_list, []):
                if attr.get('key') == "_datadump_origin_name":
                    op_names.append(attr.get('value').get('s'))

        return op_names

    def _save_dump_data(self, out, output_tensors):
        for data, tensor in zip(out, output_tensors):
            tensor_name = tensor.name.replace("/", "_").replace(":", ".") + "." + str(int(time.time()))
            npy_file_path = os.path.join(self.dump_data_npu, tensor_name)
            np.save(npy_file_path, data)
