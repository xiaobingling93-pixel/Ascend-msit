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
import os

import numpy as np
import tensorflow as tf
import tfdbg_ascend as tfdbg
from msquickcmp.common import utils, tf_common
from msquickcmp.common.dump_data import DumpData
from msquickcmp.common.utils import AccuracyCompareException

class TfSaveModelDumpData(DumpData):
    """
    This class is used to generate GUP dump data of the tf2.6 save_model.
    """

    def __init__(self, arguments):
        super().__init__()
        self.args = arguments
        output_path = os.path.realpath(self.args.out_path)
        self.input = os.path.join(output_path, "input")
        self.dump_data_tf = os.path.join(output_path, "dump_data/tf")
        self.model_name = None
        self.global_graph = None
        self.inputs_data = None
        self.model_path = self.args.model_path
        self.input_path = self.args.input_path
        self.net_output = {}
        self._load_graph()
        self._create_dir()

    def generate_inputs_data(self, npu_dump_path, om_parser=None):
        """
        Generate tf2.6 save_model inputs data
        :return tf2.6 save_model inputs data directory
        """
        inputs_tensor = tf_common.get_inputs_tensor(self.global_graph, self.args.input_shape)
        self._make_inputs_data(inputs_tensor)
        self.model_name = os.path.basename(os.path.abspath(os.path.join(npu_dump_path, "..", "..")))

    def get_net_output_info(self):
        """
        Compatible with ONNX scenarios
        """
        return self.net_output

    def generate_dump_data(self, npu_dump_path, om_parser=None):
        """
        Generate tf2.6 save_model dump data
        :return tf2.6 save_model dump data directory
        """
        model = tf.keras.models.load_model(self.model_path)
        model = compile(optimize='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # enable the dump function
        tfdbg.enable()
        tfdbg.set_dump_path(self.dump_data_tf)
        model.predict(self.inputs_data)

        self._rename_ops_name()
        return self.dump_data_tf

    def _rename_ops(self):
        ops_dump_data_dir = self.dump_data_tf
        remove_field = self.model_name
        for filename in os.listdir(ops_dump_data_dir):
            if remove_field in filename:
                new_filename = filename.replace(remove_field, '')
                old_file = os.path.join(ops_dump_data_dir, filename)
                new_file = os.path.join(ops_dump_data_dir, new_filename)
                os.rename(old_file, new_file)

    def _make_inputs_data(self, inputs_tensor):
        if self.args.input_path == "":
            if os.listdir(self.input):
                input_path = self.input
                self.input_path = ','.join([os.path.join(input_path, ii) for ii in os.listdir(input_path)])
                return

            input_path_list = []
            for index, tensor in enumerate(inputs_tensor):
                if not tensor.shape:
                    utils.logger.error(
                        "The shape of %s is unknown. Please usr -i to assign the input path." % tensor.name)
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
                input_data = np.random.random(tf_common.convert_tensor_shape(tensor.shape)) \
                    .astype(tf_common.convert_to_numpy_type(tensor.dtype))
                input_path = os.path.join(self.input, "input_" + str(index) + ".bin")
                input_path_list.append(input_path)
                try:
                    input_data.tofile(input_path)
                except Exception as err:
                    utils.logger.error("Failed to generate data %s. %s" % (input_path, err))
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR) from err
                utils.logger.info("file name: {}, shape: {}, dtype: {}".format(
                    input_path, input_data.shape, input_data.dtype))
                self.input_path = ','.join(input_path_list)
        else:
            input_path = self.args.input_path.split(",")
            if len(inputs_tensor) != len(input_path):
                utils.logger.error("the number of model inputs tensor is not equal the number of "
                                   "inputs data, inputs tensor is: {}, inputs data is: {}".format(
                    len(inputs_tensor), len(input_path)))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

    def _load_graph(self):
        self.global_graph = tf.compat.v1.get_default_graph()

    def _create_dir(self):
        # create input directory
        utils.create_directory(self.input)

        # create dump_data/tf directory
        utils.create_directory(self.input)
