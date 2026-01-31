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


class CmpArgsAdapter:
    def __init__(self,
                 gold_model,
                 om_model,
                 weight_path="",
                 input_data_path="",
                 cann_path=CANN_PATH,
                 out_path="./",
                 input_shape="",
                 device="0",
                 output_size="",
                 output_nodes="",
                 advisor=False,
                 dym_shape_range="",
                 dump=True,
                 bin2npy=False,
                 custom_op="",
                 locat=False,
                 onnx_fusion_switch=True,
                 single_op=False,
                 fusion_switch_file="",
                 max_cmp_size=0,
                 quant_fusion_rule_file="",
                 saved_model_signature="",
                 saved_model_tag_set="",
                 my_path="",
                 golden_path="",
                 ops_json=""
                 ):
        self.model_path = gold_model
        self.offline_model_path = om_model
        self.weight_path = weight_path
        self.input_path = input_data_path
        self.cann_path = cann_path
        self.out_path = out_path
        self.input_shape = input_shape
        self.device = device
        self.output_size = output_size
        self.output_nodes = output_nodes
        self.advisor = advisor
        self.dym_shape_range = dym_shape_range
        self.dump = dump
        self.bin2npy = bin2npy
        self.custom_op = custom_op
        self.locat = locat
        self.onnx_fusion_switch = onnx_fusion_switch
        self.fusion_switch_file = fusion_switch_file
        self.single_op = single_op
        self.max_cmp_size = max_cmp_size
        self.quant_fusion_rule_file = quant_fusion_rule_file
        self.saved_model_signature = saved_model_signature
        self.saved_model_tag_set = saved_model_tag_set
        self.my_path = my_path
        self.golden_path = golden_path
        self.ops_json = ops_json