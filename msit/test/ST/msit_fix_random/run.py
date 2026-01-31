# -*- coding: utf-8 -*-
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
import argparse
import logging
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from msit_llm import seed_all


def get_model_output(seed, config):
    """Helper function to get model output for a given seed and config."""
    seed_all(seed=seed)
    model = LlamaForCausalLM(config).eval()
    output = model(torch.arange(32)[None].long())
    return output.logits


def main():
    config = LlamaConfig()
    config.num_hidden_layers = 2
    config.hidden_size = 1024
    config.intermediate_size = 4096

    # Test same seed
    output1 = get_model_output(seed=1, config=config)
    output2 = get_model_output(seed=1, config=config)

    if not torch.equal(output1, output2):
        logging.error("FAIL: Output should be the same for the same seed")
        exit(1)

    # Test different seeds
    output1 = get_model_output(seed=1, config=config)
    output2 = get_model_output(seed=2, config=config)

    if torch.equal(output1, output2):
        logging.error("FAIL: Output should be different for different seeds")
        exit(1)
    
    logging.info("PASS: All tests passed")
    exit(0)


if __name__ == "__main__":
    main()
