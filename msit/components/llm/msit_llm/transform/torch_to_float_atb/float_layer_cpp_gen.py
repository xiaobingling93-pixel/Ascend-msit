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


def float_layer_cpp_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import float_layer_cpp_templates

    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += float_layer_cpp_templates.CPP_COPYRIGHT_HEADER.format(
        year=time.localtime().tm_year,
        licenses_url=get_public_url('msit_licenses_url')
    ) + "\n"
    rr += "\n".join([f'''#include "{i}"''' for i in float_layer_cpp_templates.all_atb_operation_headers])
    rr += f'''\n#include "models/{model_name_lower}/layer/decoder_layer.h"'''

    layer_core_components = float_layer_cpp_templates.DECODER_LAYER_FORMATTER.format(
        attention_formatter=float_layer_cpp_templates.ATTENTION_FORMATTER.format(),
        residual_add_formatter=float_layer_cpp_templates.RESIDUAL_ADD_FORMATTER.format(),
        mlp_formatter=float_layer_cpp_templates.MLP_FORMATTER.format(),
        mlp_residual_add_formatter=float_layer_cpp_templates.MLP_RESIDUAL_ADD_FORMATTER.format()
    )

    post_properties = "\n".join([
        float_layer_cpp_templates.PARSE_PARAM_FORMATTER.format(),
        float_layer_cpp_templates.BIND_PARAM_HOST_TENSOR_FORMATTER.format(),
    ])

    check_if_safe_string(model_name_lower)
    rr += float_layer_cpp_templates.BASIC_CLASS_FORMATTER.format(
        model_name_lower=model_name_lower,
        decoder_layer=layer_core_components,
        post_properties=post_properties,
    )

    save_name = utils.init_save_name(save_name if save_name else "decoder_layer") + ".cpp"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="layer")
    save_path = os.path.join(save_dir, save_name)
    write_file(save_path, rr)
    return save_path, rr
