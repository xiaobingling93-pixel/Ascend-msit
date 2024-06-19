# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
import torch
from torch.nn import functional as F

from ait_llm.common.log import logger


FLOAT_EPSILON = torch.finfo(torch.float).eps
NAN = 'NaN'


def cosine_similarity(golden_data: torch.Tensor, my_data: torch.Tensor):
    if torch.all(golden_data == 0) and torch.all(my_data == 0):
        return 1.0, ''  # both are all 0, return similarity 1

    result = torch.cosine_similarity(golden_data.double(), my_data.double(), dim=0).item()  # Torch handle zero data
    return round(result, 10), ''  # Trunc to keeping only 10 decimals


def max_relative_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data / golden_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        torch.tensor(0, dtype=golden_data.dtype),
    ).max()
    return result.item(), ''


def mean_relative_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data / golden_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        torch.tensor(0, dtype=my_data.dtype),
    ).mean()
    return result.item(), ''


def max_absolute_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data - golden_data),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        torch.tensor(0, dtype=my_data.dtype),
    ).max()
    return result.item(), ''


def mean_absolute_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data - golden_data),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        torch.tensor(0, dtype=my_data.dtype),
    ).mean()
    return result.item(), ''


def kl_divergence(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = F.kl_div(F.log_softmax(my_data, dim=-1), F.softmax(golden_data, dim=-1), reduction="sum").item()
    return max(result, 0), ""


def relative_euclidean_distance(golden_data: torch.Tensor, my_data: torch.Tensor):
    ground_truth_square_num = (golden_data ** 2).sum()
    if ground_truth_square_num ** 0.5 <= FLOAT_EPSILON:
        return 0.0, ''

    result = ((my_data - golden_data) ** 2).sum() / ground_truth_square_num
    return torch.sqrt(result).item(), ''


def register_custom_compare_algorithm(custom_compare_algorithm):
    import os
    import sys
    import importlib
    import inspect
    from components.utils.file_open_check import FileStat

    custom_compare_algorithm_split = custom_compare_algorithm.split(':')
    if len(custom_compare_algorithm_split) != 2:
        raise ValueError("custom_compare_algorithm should be in format '{python_file_path}:{function_name}'")
    file_path, func_name = custom_compare_algorithm_split
    file_path = os.path.expanduser(file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"custom_compare_algorithm specified {file_path} not exists")
    if not file_path.endswith(".py"):
        raise ValueError("custom_compare_algorithm specified {file_path} is not a py file")
    
    file_stat = FileStat(file_path)
    if not file_stat.is_basically_legal('read', strict_permission=False):
        raise ValueError(f"custom_compare_algorithm specified {file_path} permission stat is illegal")

    file_dir, file_name = os.path.dirname(file_path), os.path.basename(file_path)
    if len(file_dir) > 0 and file_dir not in sys.path:
        sys.path.append(file_dir)

    custom_module_name = file_name.replace('.py', '')
    try:
        custom_module = importlib.import_module(custom_module_name)
    except Exception as ee:
        raise ValueError(f"import {custom_module_name} from {file_dir} failed") from ee

    custom_compare_func = getattr(custom_module, func_name, None)
    if custom_compare_func is None:
        raise ValueError(f"getting {func_name} from {custom_compare_algorithm} failed")
    if len(inspect.signature(custom_compare_func).parameters) != 2:
        raise ValueError(f"function {func_name} signature should have exact two parameters")

    try:
        ret = custom_compare_func(torch.ones([1]), torch.ones([1]))
    except Exception as ee:
        raise ValueError(f"function {func_name} should recieve 2 torch tensor parameters")

    if not isinstance(ret, (list, tuple)) or len(ret) != 2:
        raise ValueError(f"function {func_name} should return 2 value in type ((float, int, str), str)")
    if not isinstance(ret[0], (float, int, str)) or not isinstance(ret[1], str):
        raise ValueError(f"function {func_name} should return 2 value in type ((float, int, str), str)")

    logger.info(f"Added custom comparing algorithm: {func_name}")
    CUSTOM_ALG_MAP[func_name] = custom_compare_func


CMP_ALG_MAP = {
    "cosine_similarity": cosine_similarity,
    "max_relative_error": max_relative_error,
    "mean_relative_error": mean_relative_error,
    "max_absolute_error": max_absolute_error,
    "mean_absolute_error": mean_absolute_error,
    "kl_divergence": kl_divergence,
    "relative_euclidean_distance": relative_euclidean_distance,
}

CUSTOM_ALG_MAP = {}
