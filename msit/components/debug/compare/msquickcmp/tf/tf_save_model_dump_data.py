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
        self.inputs_data = []
        self.model_path = self.args.model_path
        self.input_shape = self.args.input_shape.split(":")[1]
        self.net_output = {}
        self._create_dir()

    def generate_inputs_data(self, npu_dump_path, om_parser=None):
        """
        Generate tf2.6 save_model inputs data
        :return tf2.6 save_model inputs data directory
        """
        input_bin_data = [np.fromfile(os.path.join(self.input, input_bin_file), dtype=np.float32)
                          for input_bin_file in os.listdir(self.input)]
        for bin_data in input_bin_data:
            str_shape_list = self.input_shape.split(",")
            int_shape_list = [int(num) for num in str_shape_list]
            self.inputs_data.append(np.array(bin_data, dtype=np.float32).reshape(int_shape_list))
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
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # enable the dump function
        tfdbg.enable()
        current_dir = os.getcwd()
        os.chdir(self.dump_data_tf)
        # tfdbg.set_dump_path(self.dump_data_tf)
        model.predict(self.inputs_data)
        os.chdir(current_dir)
        self._rename_ops()
        return self.dump_data_tf

    def _rename_ops(self):
        ops_dump_data_dir = self.dump_data_tf
        remove_field = self.model_name
        for filename in os.listdir(ops_dump_data_dir):
            if remove_field in filename:
                new_filename = filename.replace(remove_field + "_", '')
                old_file = os.path.join(ops_dump_data_dir, filename)
                new_file = os.path.join(ops_dump_data_dir, new_filename)
                os.rename(old_file, new_file)

    def _create_dir(self):
        # create input directory
        utils.create_directory(self.input)

        # create dump_data/tf directory
        utils.create_directory(self.dump_data_tf)
