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

def float_layer_h_gen(parsed_model, save_name=None, save_dir=None):
    from ait_llm.transform.torch_to_float_atb import float_layer_h_templates
    
    model_name_lower = parsed_model.get("name", "model").lower()

    rr = ""
    rr += float_layer_h_templates.copyright_header.format(year=time.localtime().tm_year)
    rr += float_layer_h_templates.include_header_formater.format(
        model_name_upper=model_name_lower.upper(),
    )

    rr += float_layer_h_templates.basic_class_formatter.format(
        model_name_lower=model_name_lower,
        struct_param_formatter=float_layer_h_templates.struct_param_formatter.format(),
        decoder_layer_tensor_id_formatter=float_layer_h_templates.decoder_layer_tensor_id_formatter.format()
    )

    save_name = utils.init_save_name(save_name if save_name else "decoder_layer") + ".h"
    save_dir = utils.init_save_dir(model_name_lower if save_dir is None else save_dir, sub_dir="layer")
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as ff:
        ff.write(rr)
    return save_path, rr
