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
import os

CANN_PATH = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")


class DumpArgsAdapter:
    def __init__(self,
                 model,
                 weight_path="",
                 input_data_path="",
                 cann_path=CANN_PATH,
                 out_path="./",
                 input_shape="",
                 device="0",
                 output_size="",
                 output_nodes="",
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
                 device_pattern="",
                 om_dump_data_path="",
                 om_net_output_data_path="",
                 tf_ops_json_path="",
                 om_json_path="",
                 use_aipp_npu_dump_data_path="",
                 use_aipp_npu_net_output_data_path=""
                 ):
        self.model = model
        self.weight_path = weight_path
        self.input_path = input_data_path
        self.cann_path = cann_path
        self.out_path = out_path
        self.input_shape = input_shape
        self.device = device
        self.output_size = output_size
        self.output_nodes = output_nodes
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
        self.device_pattern = device_pattern
        self.om_dump_data_path = om_dump_data_path
        self.om_net_output_data_path = om_net_output_data_path
        self.tf_ops_json_path = tf_ops_json_path
        self.om_json_path = om_json_path
        self.use_aipp_npu_dump_data_path = use_aipp_npu_dump_data_path
        self.use_aipp_npu_net_output_data_path = use_aipp_npu_net_output_data_path
