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
COPYRIGHT_FORMATER = """# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          {licenses_url}
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""

IMPORT_FORMATER = """
import importlib
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig
from atb_llm.models.base.router import BaseRouter
"""

CLASS_ROUTER_FORMATER = """
@dataclass
class {model_name_capital}Router(BaseRouter):
    def get_config(self):
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
        )

        if not hasattr(config, 'max_position_embeddings'):
            if hasattr(config, 'max_seq_len'):
                setattr(config, 'max_position_embeddings', config.max_seq_len)
            else:
                setattr(config, 'max_position_embeddings', 4096)
        config.seq_length = config.max_position_embeddings

        
        if not hasattr(config, 'num_layers'):
            if hasattr(config, 'num_hidden_layers'):
                setattr(config, 'num_layers', config.num_hidden_layers)
        config.num_hidden_layers = config.num_layers

        if not hasattr(config, 'num_key_value_heads'):
            config.num_key_value_heads = config.num_attention_heads

        if not hasattr(config, 'rms_norm_eps'):
            if hasattr(config, 'layernorm_epsilon'):
                setattr(config, 'rms_norm_eps', config.layernorm_epsilon)
            else:
                setattr(config, 'rms_norm_eps', 1e-6)        

        # config.pe_type = 


        default_config = {{
            'pe_type': "{pe_type}",
            'max_position_embeddings': 4096,
            'model_max_length': 4096,
            'alibi_bias_max': 8,
            'rope_keep_local_base_windows': None,
            'rope_vanilla_theta': None,
            'rope_mscale': 1,
            'rope_ratio': 1,
            'rope_given_inv_feq_str': None,
            'multi_query_group_num': 2,
        }}

        for key, value in default_config.items():
            if not hasattr(self, key):
                config.__setattr__(key, value)

        super().check_config(config)
        
        return config

    def get_config_cls(self):
        model_file_dir_name = f"{{self.model_type}}."
        if self.model_version:
            model_file_dir_name = model_file_dir_name + \
                f"{{self.model_version}}."
        config_file_name = 'config'
        module_path = f"{{model_file_dir_name}}{{config_file_name}}"
        module = importlib.import_module(module_path)
        config_cls_name = f"{{self.model_type_cap}}Config"
        return getattr(module, config_cls_name)

    def get_model_cls(self):
        model_file_dir_name = f"{{self.model_type}}."
        if self.model_version:
            model_file_dir_name = model_file_dir_name + \
                f"{{self.model_version}}."
        model_file_name = 'flash_causal' if self.is_flash_causal_lm else 'causal'
        module_path = f"{{model_file_dir_name}}{{model_file_name}}_{{self.model_type}}"
        module = importlib.import_module(module_path)
        model_cls_name = f"{{self.model_type_cap}}ForCausalLM"
        if self.is_flash_causal_lm:
            model_cls_name = "Flash" + model_cls_name
        return getattr(module, model_cls_name)
"""