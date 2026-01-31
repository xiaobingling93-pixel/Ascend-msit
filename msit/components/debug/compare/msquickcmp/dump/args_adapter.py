# -*- coding: utf-8 -*-
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

CANN_PATH = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")


class DumpArgsAdapter:
    def __init__(self,
                 model_path,
                 weight_path="",
                 input_data_path="",
                 cann_path=CANN_PATH,
                 out_path="./",
                 input_shape="",
                 device="0",
                 dym_shape_range="",
                 onnx_fusion_switch=True,
                 saved_model_signature="",
                 saved_model_tag_set="",
                 device_pattern="",
                 tf_json_path="",
                 fusion_switch_file="",
                 custom_op="",
                 dump=True,
                 single_op="",
                 output_nodes=""
                 ):
        self.model_path = model_path
        self.weight_path = weight_path
        self.input_path = input_data_path
        self.cann_path = cann_path
        self.out_path = out_path
        self.input_shape = input_shape
        self.device = device
        self.dym_shape_range = dym_shape_range
        self.onnx_fusion_switch = onnx_fusion_switch
        self.saved_model_signature = saved_model_signature
        self.saved_model_tag_set = saved_model_tag_set
        self.device_pattern = device_pattern
        self.tf_json_path = tf_json_path
        self.fusion_switch_file = fusion_switch_file
        self.custom_op = custom_op
        self.dump = dump
        self.single_op = single_op
        self.output_nodes = output_nodes
