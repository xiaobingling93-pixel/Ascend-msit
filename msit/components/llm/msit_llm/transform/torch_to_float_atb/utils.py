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
import torch.nn as nn

from msit_llm.transform.utils import write_file
from components.utils.security_check import ms_makedirs


def init_save_name(save_name):
    if os.path.splitext(save_name)[-1] in [".c", ".cpp", ".h", ".hpp"]:
        save_name = os.path.splitext(save_name)[0]
    return os.path.basename(save_name)


def init_save_dir(save_dir, sub_dir):
    save_dir = os.path.abspath(save_dir)
    if os.path.basename(save_dir) in ["model", "layer"]:
        save_dir = os.path.dirname(save_dir)
    save_dir = os.path.join(save_dir, sub_dir)
    
    if not os.path.exists(save_dir):
        ms_makedirs(save_dir)
    return save_dir


def collect_module_layers(model, module_layers):
    """
    递归地将模型所有子模块及其名称存储在列表中
    如果遇到nn.ModuleList，则将其作为一个整体存储

    ：param model：nn.Module，要遍历的模型
    ：param module_layers。用于存储模型层
    """
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            module_layers.append({"type": "ModuleList", "count": len(module)})
            continue
        else:
            module_layers.append({"module": module, "type": type(module)})

        if hasattr(module, 'named_children') and callable(module.named_children):
            collect_module_layers(module, module_layers)


def get_repeat_box_layer(model):
    module_layers = []

    collect_module_layers(model, module_layers)

    res = {}
    repeat_index = 1
    
    for i in range(len(module_layers) - 1):
        if module_layers[i]["type"] != "ModuleList" and module_layers[i + 1]["type"] == "ModuleList":
            res[module_layers[i]["module"]] = {
                "repeat_type": "start",
                "repeat_count": module_layers[i + 1]["count"],
                "repeat_index": repeat_index
            }
        if module_layers[i]["type"] == "ModuleList" and module_layers[i + 1]["type"] != "ModuleList":
            res[module_layers[i + 1]["module"]] = {
                "repeat_type": "end",
                "repeat_count": module_layers[i]["count"],
                "repeat_index": repeat_index
            }
            repeat_index += 1

    return res


def dag_to_model(dag_node, is_repeat, model_layers):
    from msit_llm.transform.model_parser import parser
    parsed_model_layers = []
    for node in dag_node.dag_node_list:
        if "forward" in node.name:
            continue
        tmp_node = {}
        if isinstance(node.node, nn.Module):
            tmp_node = parser.build_model_tree(node.node)
        else:
            tmp_node['input_param'] = str(node.input_param)
        tmp_node["input_node"] = ",".join((x.name for x in node.input_nodes))
        tmp_node['name'] = node.name
        if not is_repeat:
            parsed_model_layers.append(tmp_node)
            continue
        if len(parsed_model_layers) > 0 and "repeat_block" in parsed_model_layers[-1]:
            parsed_model_layers[-1]["repeat_block"].append(tmp_node)
        else:
            parsed_model_layers.append(tmp_node)

        if node.node in model_layers:
            if model_layers[node.node]["repeat_type"] == "start":
                parsed_model_layers.append({
                    "kind": "Layers",
                    "repeat_count": model_layers[node.node]["repeat_count"],
                    "repeat_block": []
                })
            else:
                parsed_model_layers[-1]["repeat_block"].pop()
                parsed_model_layers.append(tmp_node)

    return parsed_model_layers
