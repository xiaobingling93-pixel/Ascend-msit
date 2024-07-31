# !/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
This class is used to generate GUP dump data of the tf2.6 save_model.
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from msquickcmp.common import utils, tf_common
from msquickcmp.common.dump_data import DumpData


class TfSaveModelDumpData(DumpData):
    """
    This class is used to generate GUP dump data of the tf2.6 save_model.
    """

    def __init__(self, arguments):
        super().__init__()
        output_path = os.path.realpath(arguments.out_path)
        self.input = os.path.join(output_path, "input")
        self.dump_data_tf = os.path.join(output_path, "dump_data", "tf")
        self.inputs_data = []
        self.model_path = arguments.model_path
        self.input_shape = arguments.input_shape.split(":")[-1]
        self.net_output = {}
        self._check_tf_version()
        self._create_dir()

    def generate_inputs_data(self, npu_dump_path=None, om_parser=None):
        """
        Generate tf2.6 save_model inputs data
        :return tf2.6 save_model inputs data directory
        """
        input_bin_data = [np.fromfile(os.path.join(self.input, input_bin_file), dtype=np.float32)
                          for input_bin_file in os.listdir(self.input)]
        for bin_data in input_bin_data:
            str_shape_list = self.input_shape.split(",")
            int_shape_list = [int(num) for num in str_shape_list]
            self.inputs_data.append(bin_data.reshape(int_shape_list))

    def get_net_output_info(self):
        """
        Compatible with ONNX scenarios
        """
        return self.net_output

    def generate_dump_data(self, output_json_path, npu_dump_path=None, om_parser=None):
        """
        Generate tf2.6 save_model dump data
        :return tf2.6 save_model dump data directory
        """
        ops_name = self._parse_ops_name_from_om_json(output_json_path)
        sess = tf.compat.v1.keras.backend.get_session()
        _ = tf.compat.v1.saved_model.load(sess, {'serve'}, self.model_path)
        if not self.inputs_data:
            raise ValueError("input_data is empty")
        feed_dict = {sess.graph.get_tensor_by_name('input_1:0'): self.inputs_data[0]}
        output_tensors = []
        operations = sess.graph.get_operations()
        for op_name in ops_name:
            if self._is_exists(op_name, operations):
                output_tensors.extend(sess.graph.get_operation_by_name(op_name).outputs)

        out = sess.run(output_tensors, feed_dict)
        self._save_dump_data(out, output_tensors)
        utils.logger.info("Dump tf data success, data saved in: %s", self.dump_data_tf)

        return self.dump_data_tf

    @staticmethod
    def _is_exists(operation_name_to_check, operations):
        return any(op.name == operation_name_to_check for op in operations)

    def _save_dump_data(self, out, output_tensors):
        for data, tensor in zip(out, output_tensors):
            tensor_name = tensor.name.replace("/", "_").replace(":", ".") + "." + str(int(time.time()))
            npy_file_path = os.path.join(self.dump_data_tf, tensor_name)
            np.save(npy_file_path, data)

    def _parse_ops_name_from_om_json(self, output_json_path):
        ops_name = []
        om_json = self._parse_json_file(output_json_path)
        graph_list = om_json.get('graph')
        for graph in graph_list:
            ops = graph.get('op')
            for op in ops:
                if op.get('output_desc') is not None:
                    output_desc = op.get('output_desc')
                    for od in output_desc:
                        attrs = od.get('attr')
                        for attr in attrs:
                            if attr.get('key') == "_datadump_origin_name":
                                ops_name.append(attr.get('value').get('s'))

        return ops_name

    def _create_dir(self):
        utils.create_directory(self.input)
        utils.create_directory(self.dump_data_tf)

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
    def _check_tf_version():
        expected_version = "2.6.5"
        current_version = tf.__version__
        if current_version != expected_version:
            raise ImportError(
                f"TensorFlow version mismatch: expected version {expected_version}, "
                f"but found version {current_version}. Please install the correct "
                "version of TensorFlow."
            )
