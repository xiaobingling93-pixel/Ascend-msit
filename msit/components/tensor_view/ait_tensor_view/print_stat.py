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
from tabulate import tabulate
from components.utils.log import logger


VALID_DTYPES = [
    torch.float,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.complex32,
    torch.complex64,
    torch.complex128
]


def print_stat(tensor: torch.Tensor):
    tmp = tensor
    if tensor.dtype not in VALID_DTYPES:
        tmp = tensor.clone().to(torch.float32)

    table = [
        ["min", "max", "mean", "std", "var"],
        [tmp.min(), tmp.max(), tmp.mean(), tmp.std(), tmp.var()]
    ]

    logger.info("\n%s", tabulate(table, tablefmt="grid"))
    