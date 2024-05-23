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

from json import dump

import torch.nn as nn

from kind import mlp, attention, convert, mname


def add_child(arr, node):
    arr["children"].append(node)


def filter_dropout_module(module: nn.Module):
    ret = []
    children = list(module.children())

    for c in children:
        sub_children = list(c.children())
        if ((len(sub_children) > 0 and isinstance(sub_children[0], nn.Dropout))
                or isinstance(module, nn.Dropout)):
            continue
        ret.append(c)

    return ret


def find_duplicate(modules):
    reprs = [repr(item) for item in modules]

    count = 1
    block = reprs[0]

    for r in reprs[1:]:
        if r == block:
            count += 1

    return count, modules[0]


def process_layer(layer: nn.Module):
    ret = {}

    for child in layer.children():
        lowered_name = mname(child).lower()
        sub = filter_dropout_module(child)
        size = len(sub)

        if size > 0:
            if "mlp" in lowered_name:
                ret["mlp"] = mlp(sub)
            elif "attention" in lowered_name:
                ret["attention"] = attention(sub, size)
            else:
                continue
        else:
            if "input_layernorm" in ret:
                ret["post_attention_layernorm"] = convert(child)
            else:
                ret["input_layernorm"] = convert(child)

    return ret


def build_model_tree(module: nn.Module):
    if not isinstance(module, nn.Module):
        raise ValueError("input should be torch.nn.Module")

    root = {"name": mname(module), "children": []}
    stack = [(root, module)]

    while stack:
        parent, current = stack.pop()

        for name, child in current.named_children():
            if isinstance(child, nn.ModuleList) or isinstance(child, nn.Sequential):
                repeat_count, layer = find_duplicate(child)
                repeat_block = process_layer(layer)

                root["repeat_count"] = repeat_count
                root["repeat_block"] = repeat_block
            else:
                child_node = convert(child)
                add_child(parent, child_node)
                stack.append((child_node, child))

    return root


def model_to_json(model: nn.Module, name: str):
    with open(f"{name}.json", "w") as o:
        dump(build_model_tree(model), o)
