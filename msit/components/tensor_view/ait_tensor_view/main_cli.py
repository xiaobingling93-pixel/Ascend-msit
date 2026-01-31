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
from functools import partial

from ait_tensor_view.operation import SliceOperation, PermuteOperation
from ait_tensor_view.handler import handle_tensor_view
from components.utils.parser import BaseCommand
from components.utils.file_open_check import FileStat
from components.utils.util import load_file_to_read_common_check_for_cli


def check_output_path_legality(value):
    if not value:
        return value
    try:
        file_stat = FileStat(value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"output path:{value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError(f"output path:{value} can not write. Please check.")
    return value


def parse_operations(value):
    operations = value.split(";")

    ops = []

    for op in operations:
        if op.startswith("[") and op.endswith("]"):
            ops.append(SliceOperation(op))
        elif op.startswith("(") and op.endswith(")"):
            ops.append(PermuteOperation(op))
        else:
            raise SyntaxError(f"Invalid operation string: {op}")

    return ops


def get_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


class TensorViewCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "--bin", "-b",
            type=partial(
                    load_file_to_read_common_check_for_cli,
                    exts=['.bin', '.pth']
                 ),
            required=True,
            help="Bin file path"
        )

        parser.add_argument(
            "--print", "-p",
            action="store_true",
            help="print tensor"
        )

        parser.add_argument(
            "--output", "-o",
            type=check_output_path_legality,
            help="where the tensor should be saved (Default None means current directory)"
        )

        parser.add_argument(
            "--operations", "-op",
            type=parse_operations,
            help="Each operation is separated by a semicolon; slice operations should use square brackets; permute "
                 "sequences should use parentheses"
        )

    def handle(self, args):
        handle_tensor_view(args)


def get_cmd_instance():
    help_info = "view / slice / permute / save the dumped tensor"
    cmd_instance = TensorViewCommand("tensor-view", help_info)
    return cmd_instance
