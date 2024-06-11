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
"""Quant llama2 7b to w8a16."""
import os
import argparse
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import mindspore as ms
from mindspore import log as logger
from mindformers import LlamaForCausalLM, LlamaTokenizer
from mindspore import Model
from mindspore.communication import get_rank
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import BackendTarget
from Mindspore.quant.ptq_quant.quant_config import QuantConfig
from Mindspore.quant.ptq_quant.quant_tools import Calibrator
from Mindspore.quant.ptq_quant.llm_ptq_utils import gen_fake_inputs


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--fp_ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--save_ckpt_path', '-s', type=str, required=True)
    parser.add_argument('--framework', '-f', type=str, required=True)
    parser.add_argument('--network', '-n', type=str, default="llama2_7b",
                        help="optional: llama2_7b, llama2_13b, llama2_57b, llama2_70b, baichuan2_13b, qwen_14b.")
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    print('------------------------- Creating network...', flush=True)
    quant_config = QuantConfig(uargs.config_path, uargs.network, uargs.framework, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    network = LlamaForCausalLM(quant_config.cfg.model.model_config)
    network.set_train(False)
    network.phase = 'predict'
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    start = time.time()
    rank_id = 0
    logger.info(f'Load ckpt :{uargs.fp_ckpt_path}.')
    ms.load_checkpoint(uargs.fp_ckpt_path, network)
    ms.ms_memory_recycle()
    logger.info(f'Load ckpt cost time is {time.time() - start} s.')
    print('------------------------- Quantize-ing network...', flush=True)
    Calibrator = Calibrator(network, quant_config.PTQcfg)
    Calibrator.run()
    print('------------------------- Saving checkpoint...', flush=True)
    save_path = os.path.join(uargs.save_ckpt_path, f"rank_{rank_id}")
    Calibrator.save(save_path)
    print('------------------------- Saving checkpoint success...', flush=True)