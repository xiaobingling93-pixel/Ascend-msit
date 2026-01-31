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
import os
import sys
import shutil
import re

from components.utils.log import logger
from msit_prof.msprof.args_adapter import MsProfArgsAdapter

PATH_MAX_LENGTH = 255


def remove_invalid_chars(msprof_cmd):
    invalid_chars = r'[`$|;&><]+'
    clean_msprof_cmd = re.sub(invalid_chars, '', msprof_cmd)
    return clean_msprof_cmd


def msprof_run_profiling(args, msprof_bin):
    bin_path = ' '.join(sys.argv).split(" ")[0]
    bin_path = bin_path.rsplit('/', 1)[0]
    msprof_cmd = (
        "{} --output={}/profiler --application=\"{} {}/{}\" --model-execution={}"
        " --sys-hardware-mem={} --sys-cpu-profiling={}"
        " --sys-profiling={} --sys-pid-profiling={} --dvpp-profiling={} "
        "--runtime-api={} --task-time={} --aicpu={}".format(
            msprof_bin,
            args.output,
            sys.executable,
            bin_path,
            args.application,
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
    )
    #非法字符过滤
    msprof_cmd = remove_invalid_chars(msprof_cmd)
    logger.info("msprof cmd:{} begin run".format(msprof_cmd))
    ret = os.system(msprof_cmd)
    if ret != 0:
        raise RuntimeError(f"msprof cmd failed, ret = {ret}")
    logger.info("msprof cmd:{} end run ret:{}".format(msprof_cmd, ret))


def args_rules(args):
    # output校验
    if args.output is not None and len(args.output) > PATH_MAX_LENGTH:
        logger.error("parameter --output length out of range. " "Please use it together with the parameter --output!\n")
        raise RuntimeError('error bad parameters --output')
    if args.output is None:
        args.output = os.getcwd()

    #application校验
    if args.application is None:
        logger.error(
            "parameter --application is required. " "Please use it together with the parameter --application!\n")
        raise RuntimeError('error bad parameters --application')
    if args.application is not None and len(args.application) > PATH_MAX_LENGTH:
        logger.error(
            "parameter --application length out of range." "Please use it together with the parameter --application!\n")
        raise RuntimeError('error bad parameters --application')
    
    #其他参数校验，只可能为on/off
    args_list = {
        'model_execution': args.model_execution,
        'sys_hardware_mem': args.sys_hardware_mem,
        'sys_cpu_profiling': args.sys_cpu_profiling,
        'sys_profiling': args.sys_profiling,
        'sys_pid_profiling': args.sys_pid_profiling,
        'dvpp_profiling': args.dvpp_profiling,
        'runtime_api': args.runtime_api,
        'task_time': args.task_time,
        'aicpu': args.aicpu
    }

    for args_name, args_value in args_list.items():
        if args_value is not None and args_value not in ['on', 'off']:
            logger.error(
                f"parameter --{args_name} is not valid. " f"Please use it together with the parameter --{args_name}!\n")
            raise RuntimeError(f'error bad parameters --{args_name}')

    return args


def msprof_process(args: MsProfArgsAdapter):
    try:
        args = args_rules(args)
    except RuntimeError:
        return 1
    msprof_bin = shutil.which('msprof')
    if msprof_bin is None or os.getenv('AIT_NO_MSPROF_MODE') == '1':
        logger.info("find no msprof continue use acl.json mode")
    else:
        try:
            msprof_run_profiling(args, msprof_bin)
        except RuntimeError:
            return 1
        return 0

    return 0
