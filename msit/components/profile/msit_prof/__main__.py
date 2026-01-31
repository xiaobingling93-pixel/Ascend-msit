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
from components.utils.util import safe_int


def check_positive_integer(value):
    ivalue = safe_int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_batchsize_valid(value):
    # default value is None
    if value is None:
        return value
    # input value no None
    else:
        return check_positive_integer(value)


def check_nonnegative_integer(value):
    ivalue = safe_int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid nonnegative int value" % value)
    return ivalue


def check_device_range_valid(value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    if ',' in value:
        ilist = [safe_int(v) for v in value.split(',')]
        for ivalue in ilist:
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError(
                    "{} of device:{} is invalid. valid value range is [{}, {}]".format(
                        ivalue, value, min_value, max_value
                    )
                )
        return ilist
    else:
        # default as single int value
        ivalue = safe_int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError(
                "device:{} is invalid. valid value range is [{}, {}]".format(ivalue, min_value, max_value)
            )
        return ivalue


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", required=True, help="Configure to run AI task files on the environment")
    parser.add_argument(
        "--output",
        default=None,
        help="The storage path for the collected profiling data,"
        " which defaults to the directory where the app is located",
    )
    parser.add_argument(
        "--model-execution", default="on", choices=["on", "off"],
        help="Control ge model execution performance data collection switch",
    )
    parser.add_argument(
        "--sys-hardware-mem",
        default="on",
        choices=["on", "off"],
        help="Control the read/write bandwidth data acquisition switch for ddr and llc",
    )
    parser.add_argument("--sys-cpu-profiling", default="off", choices=["on", "off"], help="CPU acquisition switch")
    parser.add_argument(
        "--sys-profiling",
        default="off",
        choices=["on", "off"],
        help="System CPU usage and system memory acquisition switch",
    )
    parser.add_argument(
        "--sys-pid-profiling",
        default="off",
        choices=["on", "off"],
        help="The CPU usage of the process and the memory collection switch of the process",
    )
    parser.add_argument("--dvpp-profiling", default="on", choices=["on", "off"], help="DVPP acquisition switch")

    parser.add_argument(
        "--runtime-api",
        default="on",
        choices=["on", "off"],
        help="Control runtime api performance data collection switch",
    )
    parser.add_argument(
        "--task-time",
        default="on",
        choices=["on", "off"],
        help="Control ts timeline performance data collection switch",
    )
    parser.add_argument(
        "--aicpu", default="on", choices=["on", "off"], help="Control aicpu performance data collection switch"
    )
    args_ret = parser.parse_args()

    return args_ret


if __name__ == "__main__":
    from msprof.msprof_process import msprof_process
    from msprof.args_adapter import MsProfArgsAdapter
    args = get_args()

    args = MsProfArgsAdapter(
        args.application,
        args.output,
        args.model_execution,
        args.sys_hardware_mem,
        args.sys_cpu_profiling,
        args.sys_profiling,
        args.sys_pid_profiling,
        args.dvpp_profiling,
        args.runtime_api,
        args.task_time,
        args.aicpu,
    )
    ret = msprof_process(args)
    exit(ret)
