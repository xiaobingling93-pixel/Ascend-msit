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

import os
import time
from ait_llm.transform.torch_to_float_atb import utils

def float_model_cpp_gen(parsed_model, save_name=None, save_dir=None):
    from ait_llm.transform.torch_to_float_atb import float_model_cpp_templates as templates  # avoiding circular import

    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += templates.copyright_header.format(year=time.localtime().tm_year)
    rr += templates.include_header_formater.format(
        model_name_lower=model_name_lower,
        other_operations="",
    )

    pre_properties = "\n".join([
        templates.weight_count_formatter.format(),
        templates.operation_count_formatter.format(),
        templates.in_tensor_id_formatter.format(),
        templates.internal_tensor_id_formatter.format(),
        templates.out_tensor_id_formatter.format(),
        templates.from_string_formatter.format(),
    ])

    build_graph = templates.build_graph_formatter.format(
        build_graph_pre_process_formatter=templates.build_graph_pre_process_formatter.format(),
        build_graph_pre_process_norm_formatter=templates.build_graph_pre_process_norm_formatter.format(),
        build_graph_layers_formatter=templates.build_graph_layers_formatter.format(model_name_lower=model_name_lower),
        build_graph_post_process_norm_formatter=templates.build_graph_post_process_norm_formatter.format(),
        build_graph_post_process_lmhead_formatter=templates.build_graph_post_process_lmhead_formatter.format(),
    )

    post_properties = "\n".join([
        templates.infer_shape_formatter.format(),
        build_graph,
        templates.parse_param_formatter.format(),
        templates.bind_param_host_tensor_formatter.format(),
    ])

    rr += templates.basic_class_formatter.format(
        model_name_lower=model_name_lower,
        pre_properties=pre_properties,
        post_properties=post_properties,
    )

    save_name = utils.init_save_name("decoder_model" if save_name is None else save_name) + ".cpp"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="model")
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as ff:
        ff.write(rr)
    return save_path, rr