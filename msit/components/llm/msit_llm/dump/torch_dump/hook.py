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
import torch

from msit_llm.common.log import logger
from msit_llm.dump.torch_dump.dump_config import DumpConfig
from msit_llm.dump.torch_dump.dump_hook import DumpHookModule


HOOK_TYPE = ""


def get_device(model):
    device = "cpu"

    for param in model.parameters():
        device = str(param.device)

    return device.replace(":", "")


def register_hook(model, config: DumpConfig, hook_type="dump_data"):
    global HOOK_TYPE
    if HOOK_TYPE:  # 避免重复hook
        logger.warning("%s has been register in model.", HOOK_TYPE)
        return
    else:
        HOOK_TYPE = hook_type

    if not isinstance(model, torch.nn.Module):
        logger.error("model must be instance of torch.nn.Module.")
        return

    if type(config).__name__ != "DumpConfig":
        logger.error("config must be instance of DumpConfig.")
        return

    device = get_device(model)
    config.set_dump_device_and_dump_dir(device)

    if HOOK_TYPE == "dump_data":
        hook_module = DumpHookModule(model, config)
        hook_module.add_hook()
