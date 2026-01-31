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
from os.path import splitext

import torch

from ait_tensor_view.atb import read_atb_data, write_atb_data
from ait_tensor_view.print_stat import print_stat
from components.utils.log import logger
from components.utils.util import safe_torch_load


def replace(in_path: str, out_path: str) -> str:
    in_ext = splitext(in_path)[1]
    out_ext = splitext(out_path)[1]

    if in_ext and not out_ext:
        out_path += in_ext

    return out_path


def handle_tensor_view(args):
    in_ext = splitext(args.bin)[1]
    if in_ext == ".bin":
        tensor = read_atb_data(args.bin)
    else:
        tensor = safe_torch_load(args.bin, map_location="cpu")

    logger.info(f"source tensor shape: {tensor.shape}")

    if args.operations:
        logger.info("Operations start")

        for op in args.operations:
            logger.info(f"{op.name} starts, current tensor shape: {tensor.shape}")
            tensor = op.process(tensor)
            logger.info(f"{op.name} ends, current tensor shape: {tensor.shape}")

        logger.info("Operations end")

    print_stat(tensor)

    if args.print:
        logger.info("\n%s", tensor)

    if args.output:
        out_path = args.output
        out_ext = splitext(out_path)[1]

        try:
            if not out_ext or (out_ext != ".bin" and out_ext != ".pth"):
                out_path += in_ext
            if out_path.endswith(".bin"):
                write_atb_data(tensor, out_path)
            else:
                torch.save(tensor, out_path)

            logger.info(f'Tensor saved successfully to {out_path}')
        except Exception as e:
            logger.error(f"Error saving Tensor: {e}")
