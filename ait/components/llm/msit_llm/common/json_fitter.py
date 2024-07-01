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
import sys
import json
import base64


def atb_node_to_plain_node(atb_node_dict, level, target_level):
    if target_level != -1 and level >= target_level:
        return [atb_node_dict]
    
    # 递归元
    if "nodes" in atb_node_dict:
        plain_nodes = []
        for node_dict in atb_node_dict["nodes"]:
            plain_nodes = plain_nodes + atb_node_to_plain_node(node_dict, level + 1, target_level)
        return plain_nodes
    else:
        return [atb_node_dict]


def atb_json_dict_node_parse(atb_json_dict, target_level):
    plain_atb_nodes = []

    if target_level == 0 or "nodes" not in atb_json_dict:
        return [atb_json_dict]

    else:
        raw_atb_nodes = atb_json_dict["nodes"]
        level = 1
        for node in raw_atb_nodes:
            plain_atb_nodes = plain_atb_nodes + atb_node_to_plain_node(node, level, target_level)
        return plain_atb_nodes


def atb_param_to_onnx_attribute(atb_param_name, atb_param_value):
    onnx_attr_dict = {}
    onnx_attr_dict["name"] = atb_param_name

    if isinstance(atb_param_value, str):
        onnx_attr_dict["type"] = "STRINGS"
        onnx_attr_dict["strings"] = [str(base64.b64decode(atb_param_value.encode("utf-8")), "utf-8")]
        return onnx_attr_dict

    onnx_attr_dict["type"] = "FLOATS"
    values = []
    if isinstance(atb_param_value, list):
        for v in atb_param_value:
            values.append(float(v))
    elif atb_param_value:
        values.append(float(atb_param_value))
    onnx_attr_dict["floats"] = values
    return onnx_attr_dict


def parse_onnx_attr_from_atb_node_dict(atb_node_dict):
    onnx_attrs = []

    if "param" not in atb_node_dict:
        return onnx_attrs
    
    for param_name in atb_node_dict["param"]:
        if isinstance(atb_node_dict["param"][param_name], dict):
            for sub_param_name in atb_node_dict["param"][param_name]:
                full_name = param_name + "." + sub_param_name
                onnx_attr_dict = atb_param_to_onnx_attribute(full_name, atb_node_dict["param"][param_name][sub_param_name])
                onnx_attrs.append(onnx_attr_dict)
        else:
            onnx_attr_dict = atb_param_to_onnx_attribute(param_name, atb_node_dict["param"][param_name])
        onnx_attrs.append(onnx_attr_dict)
    return onnx_attrs


def atb_node_to_onnx_node(atb_node_dict):
    onnx_node_dict = {}
    onnx_node_dict["name"] = atb_node_dict["opName"]
    onnx_node_dict["opType"] = atb_node_dict["opType"]
    onnx_node_dict["input"] = atb_node_dict["inTensors"]
    onnx_node_dict["output"] = atb_node_dict["outTensors"]
    onnx_node_dict["attribute"] = parse_onnx_attr_from_atb_node_dict(atb_node_dict)
    return onnx_node_dict


def atb_json_to_onnx_json(atb_json_dict, target_level):
    onnx_json_dict = {}
    plain_nodes = atb_json_dict_node_parse(atb_json_dict, target_level)

    for i, plain_node in enumerate(plain_nodes):
        plain_nodes[i] = atb_node_to_onnx_node(plain_node)

    onnx_json_dict["graph"] = {}
    onnx_json_dict["graph"]["node"] = plain_nodes

    onnx_json_dict["graph"]["input"] = []
    for in_tensor_name in atb_json_dict["inTensors"]:
        onnx_input_tensor_dict = {}
        onnx_input_tensor_dict["name"] = in_tensor_name
        onnx_json_dict["graph"]["input"].append(onnx_input_tensor_dict)

    onnx_json_dict["graph"]["output"] = []
    for out_tensor_name in atb_json_dict["outTensors"]:
        onnx_output_tensor_dict = {}
        onnx_output_tensor_dict["name"] = out_tensor_name
        onnx_json_dict["graph"]["output"].append(onnx_output_tensor_dict)
    return onnx_json_dict


def atb_json_to_onnx(atb_json_path, target_level=-1):
    import onnx
    from google.protobuf.json_format import Parse

    with open(atb_json_path, "r") as file:
        json_content = json.loads(file.read(), parse_constant=lambda x: None)
    
    onnx_json = atb_json_to_onnx_json(json_content, target_level)
    onnx_str = json.dumps(onnx_json)
    convert_model = Parse(onnx_str, onnx.ModelProto())
    onnx_dir = atb_json_path[0:-5] + ".onnx"
    onnx.save(convert_model, onnx_dir)

