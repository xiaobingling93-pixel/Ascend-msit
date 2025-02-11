# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import fnmatch
from typing import Optional, Set, List, Union, Dict
from abc import abstractmethod

import torch

class BaseSaver:
    def __init__(self):
        self._enable = False

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        pass

    @property
    def is_enabled(self):
        return self._enable
    
    @abstractmethod
    def post_process(self):
        pass
    
    def enable(self):
        self._enable = True
    
    def disable(self):
        self._enable = False
    

class TensorCollector:
    REGISTERED_SAVER_CLASSES = {}
    
    def __init__(self, cache_key_set: Optional[Set[str]] = None, cache_key_patterns: Optional[List[str]] = None):
        self.savers: List[BaseSaver] = []
        self.cache_key_set: Set[str] = cache_key_set if cache_key_set else set()
        self.cache_key_patterns: List[str] = cache_key_patterns if cache_key_patterns else []
        self.cached_tensors: Dict[str, torch.Tensor] = {}

    def __setitem__(self, key, value):
        for saver in self.savers:
            saver[key] = value

        if key in self.cache_key_set:
            self.cached_tensors[key] = value
                
        for pattern in self.cache_key_patterns:
            if fnmatch.fnmatch(key, pattern):
                self.cached_tensors[key] = value

    def __getitem__(self, key):
        if key in self.cached_tensors:
            return self.cached_tensors[key]

        return None
    
    @property
    def model_quant_type(self):
        return self.savers[0].model_quant_type
    
    @model_quant_type.setter
    def model_quant_type(self, value):
        for saver in self.savers:
            saver.model_quant_type = value

    @staticmethod
    def register_saver_cls(name, saver_cls):


        TensorCollector.REGISTERED_SAVER_CLASSES[name] = saver_cls

    def post_process(self):
        for saver in self.savers:
            saver.post_process()

    def clear(self):
        self.cached_tensors.clear()

    def add_saver(self, saver: Union[BaseSaver, str]):
        if isinstance(saver, str):
            if saver in self.REGISTERED_SAVER_CLASSES:
                self.savers.append(self.REGISTERED_SAVER_CLASSES[saver]())
            else:
                raise ValueError(f"Saver {saver} not found")
        else:
            self.savers.append(saver)

    def remove_saver(self, saver: BaseSaver):
        self.savers.remove(saver)


