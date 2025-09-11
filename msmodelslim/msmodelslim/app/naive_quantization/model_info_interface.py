#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from abc import ABC, abstractmethod

from msmodelslim.utils.exception import UnsupportedError


class ModelInfoInterface(ABC):
    """
    Interface for getting model information to assist best practice selection.
    """

    @abstractmethod
    def get_model_pedigree(self) -> str:
        """
        Get the pedigree of the model, e.g. "qwen3", "deepseek_v3", etc.
        Best practices categorized by pedigree in the repository.
        
        Returns:
            str: The pedigree of the model.
        """
        raise UnsupportedError(
            "This model does not support get model pedigree.",
            action="Please implement get_model_pedigree in ModelInfoInterface for naive quantization.")

    @abstractmethod
    def get_model_type(self) -> str:
        """
        Get the type of the model, e.g. "Qwen3-32B", "DeepSeek-R1", etc.
        Best practices labeled by model type, which is used to limit the best practice candidates.

        Returns:
            str: The type of the model.
        """
        raise UnsupportedError(
            "This model does not support get model type.",
            action="Please implement get_model_type in ModelInfoInterface for naive quantization.")
