# Copyright 2022 Huawei Technologies Co., Ltd
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
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.


class QuantConfig(object):
    def __init__(self, 
                 disable_names: object = None,
                 w_bit = 'int8',
                 a_bit = 'int8',
                 use_kvcache_quant=False,
                 do_smooth=False
                 ):
        
            self.disable_names=disable_names,
            self.w_bit=w_bit,
            self.a_bit=a_bit,
            self.use_kvcache_quant=use_kvcache_quant,
            self.do_smooth=do_smooth
