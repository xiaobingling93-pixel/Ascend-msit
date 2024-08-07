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
import glob
import os
import shutil

import npu_device
import numpy as np
import tensorflow as tf
from msquickcmp.atc import atc_utils
from msquickcmp.common import utils
from npu_device.compat.v1.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


class NpuTfAdapterDumpData(object):
    """
    This class is used to generate GUP dump data of the tf2.6 save_model.
    """

    def __init__(self, arguments):
        self.output_path = os.path.realpath(arguments.out_path)
        self.input = os.path.join(self.output_path, "input")
        self.dump_data_npu = os.path.join(self.output_path, "dump_data", "npu")
        self.model_json_path = os.path.join(self.output_path, "model")
        self.inputs_data = {}
        self.model_path = arguments.offline_model_path
        self.input_shape = self.split_input_shape(arguments.input_shape)
        self.cann_path = arguments.cann_path
        self._create_dir()

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
    def get_graph_txt(model_json_path):
        search_pattern = 'ge_proto*Build.txt'
        matching_files = glob.glob(os.path.join(model_json_path, search_pattern))

        return os.path.join(model_json_path, matching_files[0])

    def _create_dir(self):
        utils.create_directory(self.input)
        utils.create_directory(self.dump_data_npu)
        utils.create_directory(self.model_json_path)

    def generate_inputs_data(self):
        for shape in self.input_shape:
            input_data = np.random.random(size=shape[1]).astype(np.float32)
            self.inputs_data[shape[0]] = input_data
            input_data.tofile(os.path.join(self.input, shape[0] + ".bin"))

    def generate_dump_data(self):
        graph = self._load_graph()
        # adapt NPU
        npu_device.compat.enable_v1()
        # switch ge graph dump
        os.environ['DUMP_GE_GRAPH'] = '2'
        config_proto = tf.compat.v1.ConfigProto()
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(self.dump_data_npu)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        tf_config = npu_config_proto(config_proto=config_proto)
        # sess run predict
        with tf.compat.v1.Session(config=tf_config, graph=graph) as sess:
            feed_dict = {
                sess.graph.get_tensor_by_name(input_name + ":0"): input_data
                for input_name, input_data in self.inputs_data.items()
            }
            current_path = os.getcwd()
            os.chdir(self.model_json_path)
            sess.run(tf.no_op(), feed_dict=feed_dict)
            utils.logger.info("Dump tf adapter data success, data saved in: %s", self.dump_data_npu)
        os.chdir(current_path)
        self._move_dump_data_files()
        graph_txt = self.get_graph_txt(self.model_json_path)
        output_json_path = atc_utils.convert_model_to_json(self.cann_path, graph_txt, self.output_path)

        return self.dump_data_npu, output_json_path

    def _load_graph(self):
        sess = tf.compat.v1.keras.backend.get_session()
        tag_set = {tf.compat.v1.saved_model.tag_constants.SERVING}
        _ = tf.compat.v1.saved_model.load(sess, tag_set, self.model_path)

        return sess.graph

    def _move_dump_data_files(self):
        for item in os.listdir(self.dump_data_npu):
            item_path = os.path.join(self.dump_data_npu, item)
            if os.path.isdir(item_path):
                for file in os.listdir(item_path):
                    file_path = os.path.join(item_path, file)
                    if os.path.isfile(file_path):
                        shutil.move(file_path, os.path.join(self.dump_data_npu, file))
                shutil.rmtree(item_path)
