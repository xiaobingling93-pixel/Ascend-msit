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


def float_layer_h_gen(parsed_model, save_name=None, save_dir=None):
    from msit_llm.transform.torch_to_float_atb import float_layer_h_templates

    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += float_layer_h_templates.COPYRIGHT_HEADER.format(
        year=time.localtime().tm_year,
        licenses_url=get_public_url('msit_licenses_url')
    )

    check_if_safe_string(model_name_lower)
    rr += float_layer_h_templates.INCLUDE_HEADER_FORMATTER.format(
        model_name_upper=model_name_lower.upper(),
    )

    rr += float_layer_h_templates.BASIC_CLASS_FORMATTER.format(
        model_name_lower=model_name_lower,
        STRUCT_PARAM_FORMATTER=float_layer_h_templates.STRUCT_PARAM_FORMATTER.format(),
        DECODER_LAYER_TENSOR_ID_FORMATTER=float_layer_h_templates.DECODER_LAYER_TENSOR_ID_FORMATTER.format()
    )

    save_name = utils.init_save_name(save_name if save_name else "decoder_layer") + ".h"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="layer")
    save_path = os.path.join(save_dir, save_name)
    write_file(save_path, rr)
    return save_path, rr
