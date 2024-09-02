# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import random
import argparse
import torch
import numpy as np

try:
    import torch_npu
except ImportError:
    IS_GPU = True
else:
    IS_GPU = False

from msit_llm.common.tool import read_atb_data
from msit_llm.compare.cmp_utils import compare_data
from msit_llm.common.json_fitter import atb_json_to_onnx
from msit_llm.dump.torch_dump import DumpConfig
from msit_llm.dump.torch_dump import register_hook
from msit_llm.metrics.case_filter import CaseFilter
from msit_llm.bc_analyze import Analyzer, Synthesizer
from msit_llm.common.log import logger
from msit_llm.common.constant import LCCL_DETERMINISTIC, HCCL_DETERMINISTIC, \
    ATB_MATMUL_SHUFFLE_K_ENABLE, ATB_LLM_LCOC_ENABLE, PYTHON_HASH_SEED


def seed_all(seed=1, mode=False):
    if not isinstance(seed, int):
        raise argparse.ArgumentTypeError("%s is not an int." % seed)
    if not isinstance(mode, bool):
        raise argparse.ArgumentTypeError("%s is not a bool." % mode)
    
    os.environ[LCCL_DETERMINISTIC] = "1"
    os.environ[HCCL_DETERMINISTIC] = "1"
    os.environ[ATB_MATMUL_SHUFFLE_K_ENABLE] = "0"
    os.environ[ATB_LLM_LCOC_ENABLE] = "0"

    os.environ[PYTHON_HASH_SEED] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=mode)

    if IS_GPU and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.bachekds.cudnn.benchmark = False
    else:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

    logger.info(f"Enable deterministic computation sucess! current seed is {seed},"
                f"torch deterministic algorithms mode is {mode}.")