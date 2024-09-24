# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from mindspore import dtype as msdtype
from mindformers import MindFormerConfig

from msmodelslim.mindspore.quant.llm_ptq.mindspore_gs.ptq.network_helpers.mf_net_helpers \
    import MFLlama2Helper, MFParallelLlama2Helper
from ascend_utils.common.security import check_type, check_element_type


class QuantConfig(object):
    def __init__(self, 
                 w_bit: int = 8,
                 a_bit: int = 8,
                 use_kvcache_quant: bool = False,
                 do_smooth: bool = False,
                 disable_names: object = None,
                 config_path: str = None,
                 ):

            self.w_bit = w_bit
            self.a_bit = a_bit
            self.use_kvcache_quant = use_kvcache_quant
            self.do_smooth = do_smooth
            self.disable_names = disable_names
            self.config_path = config_path
            self._check_params()
            self.w_bit = msdtype.int8 if self.w_bit == 8 else None
            self.a_bit = msdtype.int8 if self.a_bit == 8 else None
            self.use_kvcache_quant = msdtype.int8 if self.use_kvcache_quant else None
            self.msconfig = self.create_msconfig(self.config_path)
    
    def _check_params(self):
        w_bit_list = [8, 16]
        a_bit_list = [8, 16]
        if self.w_bit not in w_bit_list:
            raise ValueError("w_bit is invalid, please check it.")
        if self.a_bit not in a_bit_list:
            raise ValueError("a_bit is invalid, please check it.")
        check_type(self.use_kvcache_quant, bool, param_name="use_kvcache_quant")
        check_type(self.do_smooth, bool, param_name="use_kvcache_quant")
        check_type(self.disable_names, list, param_name="disable_names")
        check_element_type(self.disable_names, str, param_name="disable_names")
        check_type(self.config_path, str, param_name="config_path")
    
    def create_msconfig(self, config_path):
        """Create mindformers config for llama2 network for example."""
        msconfig = MindFormerConfig(config_path)
        if msconfig.model.arch.type == "LlamaForCausalLM":
            helper = MFLlama2Helper(config_path)
        elif msconfig.model.arch.type == "ParallelLlamaForCausalLM":
            helper = MFParallelLlama2Helper(config_path)
        else:
            err_msg = f"Unsupported network arch: {msconfig.model.arch}," \
                      f"please check model.arch in yaml config, " \
                      f"only support LlamaForCausalLM and ParallelLlamaForCausalLM now"
            raise ValueError(err_msg)
        return helper