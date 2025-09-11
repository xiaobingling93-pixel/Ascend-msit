#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from abc import abstractmethod


class QuaRotAdapter:

    @abstractmethod
    def get_hidden_dim(self):
        pass

    @abstractmethod
    def get_head_dim(self):
        pass

    @abstractmethod
    def get_num_attention_heads(self):
        pass

    @abstractmethod
    def get_num_key_value_heads(self):
        pass

    @abstractmethod
    def get_lm_head(self) -> str:
        pass

    @abstractmethod
    def get_pre_head_layernorm(self) -> str:
        pass

    @abstractmethod
    def get_embedding(self) -> str:
        pass

    @abstractmethod
    def get_layer_wise_norm_liner_pair(self, decoder_module):
        pass

    @abstractmethod
    def get_layer_wise_ov_pair(self, decoder_module):
        pass

    @abstractmethod
    def get_layer_wise_up_down_pair(self, decoder_module):
        pass
