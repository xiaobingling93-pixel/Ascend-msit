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
import time

from components.utils.install import get_public_url
from msit_llm.transform.torch_to_float_atb import utils
from msit_llm.transform.utils import write_file, check_if_safe_string


def float_model_cpp_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import float_model_cpp_templates as templates  # avoiding circular import

    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += templates.COPYRIGHT_HEADER.format(
        year=time.localtime().tm_year,
        licenses_url=get_public_url('msit_licenses_url')
    )

    check_if_safe_string(model_name_lower)
    rr += templates.INCLUDE_HEADER_FORMATTER.format(
        model_name_lower=model_name_lower,
        other_operations="",
    )

    pre_properties = "\n".join([
        templates.WEIGHT_COUNT_FORMATTER.format(),
        templates.OPERATION_COUNT_FORMATTER.format(),
        templates.IN_TENSOR_ID_FORMATTER.format(),
        templates.INTERNAL_TENSOR_ID_FORMATTER.format(),
        templates.OUT_TENSOR_ID_FORMATTER.format(),
        templates.FROM_STRING_FORMATTER.format(),
    ])

    build_graph = templates.BUILD_GRAPH_FORMATTER.format(
        BUILD_GRAPH_PRE_PROCESS_FORMATTER=templates.BUILD_GRAPH_PRE_PROCESS_FORMATTER.format(),
        BUILD_GRAPH_PRE_PROCESS_NORM_FORMATTER=templates.BUILD_GRAPH_PRE_PROCESS_NORM_FORMATTER.format(),
        BUILD_GRAPH_LAYERS_FORMATTER=templates.BUILD_GRAPH_LAYERS_FORMATTER.format(model_name_lower=model_name_lower),
        BUILD_GRAPH_POST_PROCESS_NORM_FORMATTER=templates.BUILD_GRAPH_POST_PROCESS_NORM_FORMATTER.format(),
        BUILD_GRAPH_POST_PROCESS_LMHEAD_FORMATTER=templates.BUILD_GRAPH_POST_PROCESS_LMHEAD_FORMATTER.format(),
    )

    post_properties = "\n".join([
        templates.INFER_SHAPE_FORMATTER.format(),
        build_graph,
        templates.PARSE_PARAM_FORMATTER.format(),
        templates.BIND_PARAM_HOST_TENSOR_FORMATTER.format(),
    ])

    rr += templates.BASIC_CLASS_FORMATTER.format(
        model_name_lower=model_name_lower,
        pre_properties=pre_properties,
        post_properties=post_properties,
    )

    save_name = utils.init_save_name("decoder_model" if save_name is None else save_name) + ".cpp"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="model")
    save_path = os.path.join(save_dir, save_name)
    write_file(save_path, rr)
    return save_path, rr
