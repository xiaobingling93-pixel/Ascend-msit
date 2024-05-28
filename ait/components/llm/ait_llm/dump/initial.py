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


import os
import site
import subprocess
import shutil
import re

from components.utils.file_open_check import FileStat
from ait_llm.common.log import logger
from ait_llm.common.utils import safe_string
from ait_llm.common.constant import ATB_HOME_PATH, ATB_SAVE_TENSOR_TIME, ATB_SAVE_TENSOR_IDS, \
    ATB_SAVE_TENSOR_RUNNER, ATB_SAVE_TENSOR, ATB_SAVE_TENSOR_RANGE, \
    ATB_SAVE_TILING, LD_PRELOAD, ATB_OUTPUT_DIR, ATB_SAVE_CHILD, ATB_SAVE_TENSOR_PART, \
    ASCEND_TOOLKIT_HOME, ATB_PROB_LIB_WITH_ABI, ATB_PROB_LIB_WITHOUT_ABI, ATB_SAVE_CPU_PROFILING, \
    ATB_CUR_PID, ATB_DUMP_SUB_PROC_INFO_SAVE_PATH, ATB_DEVICE_ID, ATB_AIT_LOG_LEVEL, ATB_DUMP_TYPE, get_ait_dump_path


def is_use_cxx11():
    atb_home_path = os.environ.get(ATB_HOME_PATH, "")
    if not atb_home_path or not os.path.exists(atb_home_path):
        raise OSError("ATB_HOME_PATH from atb is required, but it is empty or invalid.")
    atb_home_path = safe_string(atb_home_path)
    lib_atb_path = os.path.join(atb_home_path, "lib", "libatb.so")
    if not os.path.exists(lib_atb_path):
        raise OSError(f"{lib_atb_path} not exists, please make sure atb is compiled correctly")

    result_code, abi_result = subprocess.getstatusoutput(f"nm -D {lib_atb_path} | grep Probe | grep cxx11")
    if result_code == 1 and len(abi_result) == 0:  # Execute succesfully but not found
        return False
    elif result_code != 0:
        logger.warning("Detecting abi status from atb so failed, will regard it as False")
        return False
    else:
        return len(abi_result) > 0


def init_dump_task(args):
    if args.save_desc:
        os.environ[ATB_SAVE_TENSOR] = "2"
    else:
        os.environ[ATB_SAVE_TENSOR] = "1"

    os.environ[ATB_SAVE_TENSOR_TIME] = str(args.time)
    if args.ids:
        os.environ[ATB_SAVE_TENSOR_IDS] = str(args.ids)
    else:
        os.environ.pop(ATB_SAVE_TENSOR_IDS, None)  # Ensure none is set

    if args.opname:
        os.environ[ATB_SAVE_TENSOR_RUNNER] = str(args.opname).lower()
    else:
        os.environ.pop(ATB_SAVE_TENSOR_RUNNER, None)  # Ensure none is set

    get_ait_dump_path()

    if args.output:
        if args.output.endswith('/'):
            os.environ[ATB_OUTPUT_DIR] = str(args.output)
        else:
            os.environ[ATB_OUTPUT_DIR] = str(args.output) + '/'
    else:
        os.environ.pop(ATB_OUTPUT_DIR, None)  # Ensure none is set

    if args.type:
        os.environ[ATB_DUMP_TYPE] = "|".join(args.type)
    else:
        os.environ.pop(ATB_DUMP_TYPE, None)  # Ensure none is set

    if "onnx" in args.type and ("model" in args.type or "layer" in args.type):
        os.environ[ATB_DUMP_SUB_PROC_INFO_SAVE_PATH] = os.path.join(str(args.output), str(os.getpid()))
        subprocess_info_path = os.path.join(args.output, str(os.getpid()))
        os.makedirs(subprocess_info_path, exist_ok=True)
    else:
        os.environ.pop(ATB_DUMP_SUB_PROC_INFO_SAVE_PATH, None)  # Ensure none is set

    os.environ[ATB_SAVE_CHILD] = "1" if args.child else "0"
    os.environ[ATB_SAVE_TENSOR_RANGE] = str(args.range)
    os.environ[ATB_SAVE_TILING] = "1" if args.tiling else "0"
    os.environ[ATB_SAVE_TENSOR_PART] = str(args.save_tensor_part)
    os.environ[ATB_SAVE_CPU_PROFILING] = "1" if "cpu_profiling" in args.type else "0"
    os.environ[ATB_CUR_PID] = str(os.getpid())

    if args.device_id is not None:
        os.environ[ATB_DEVICE_ID] = str(args.device_id)
    else:
        os.environ.pop(ATB_DEVICE_ID, None)  # Ensure none is set

    atb_log_level_map = {"debug": 0, "info": 1, "warning": 2, "warn": 2, "error": 3, "fatal": 4, "critical": 5}
    cur_log_level = atb_log_level_map.get(args.log_level.lower(), 1)
    os.environ[ATB_AIT_LOG_LEVEL] = str(cur_log_level)

    cann_path = os.environ.get(ASCEND_TOOLKIT_HOME, "/usr/local/Ascend/ascend-toolkit/latest")
    if not cann_path or not os.path.exists(cann_path):
        raise OSError("cann_path is invalid, please install cann-toolkit and set the environment variables.")

    cur_is_use_cxx11 = is_use_cxx11()
    logger.info(f"Info detected from ATB so is_use_cxx11: {cur_is_use_cxx11}")
    save_tensor_so_name = ATB_PROB_LIB_WITH_ABI if cur_is_use_cxx11 else ATB_PROB_LIB_WITHOUT_ABI
    save_tensor_so_path = os.path.join(cann_path, "tools", "ait_backend", "dump", save_tensor_so_name)
    if not os.path.exists(save_tensor_so_path):
        raise OSError(f"{save_tensor_so_name} is not found in {cann_path}. Try installing the latest cann-toolkit")
    if not FileStat(save_tensor_so_path).is_basically_legal('read', strict_permission=True):
        raise OSError(f"{save_tensor_so_name} is illegal, group or others writable file stat is not permitted")

    logger.info(f"Append save_tensor_so_path: {save_tensor_so_path} to LD_PRELOAD")
    ld_preload = os.getenv(LD_PRELOAD) or ""
    os.environ[LD_PRELOAD] = save_tensor_so_path + ":" + ld_preload


def json_to_onnx(args):
    subprocess_info_file = os.path.join(str(args.output), str(os.getpid()), 'subprocess_info.txt')
    if not os.path.exists(subprocess_info_file):
        return

    with open(subprocess_info_file) as f:
        from ait_llm.common.json_fitter import atb_json_to_onnx
        for line in f.readlines():
            path = line.strip()
            if not os.path.exists(path):
                continue
            atb_json_to_onnx(path)

    # clean tmp file
    subprocess_info_dir = os.path.join(args.output, str(os.getpid()))
    if os.path.isdir(subprocess_info_dir):
        shutil.rmtree(subprocess_info_dir)


def read_cpu_profiling_data(lines, data_map):
    for line in lines:
        # 解析opname和数据
        match = re.match(r'\[([a-zA-Z0-9_]*)\]:(.*)', line)
        if match:
            opname = match.group(1)
            stats = match.group(2)
            # 将数据添加到字典中
            if opname in data_map:
                data_map[opname].append(stats)
            else:
                data_map[opname] = [stats]


def split_cpu_profiling_data(data, opname):
    execute_data = ''
    setup_data = ''
    # 遍历每个opname的数据
    for stats in data[opname]:
        # 提取execute和setup数据
        execute_match = re.search(r'kernelExecuteTime:(\d+)', stats)
        setup_match = re.search(r'runnerSetupTime:(\d+)', stats)
        if execute_match:
            execute_data = stats
        elif setup_match:
            setup_data = stats
    return execute_data, setup_data


def merge_cpu_profiling_data(path):
    # 遍历目录下所有文件
    for root, dirs, files in os.walk(path):
        for file in files:
            if not re.match(r'operation_statistic_\d+\.txt', file):
                continue
            data = {}
            with open(os.path.join(root, file), 'r+') as f:
                lines = f.readlines()
                read_cpu_profiling_data(lines, data)
                f.truncate(0)
                f.seek(0, os.SEEK_SET)
                for opname in data.keys():
                    execute_data, setup_data = split_cpu_profiling_data(data, opname)
                    merged_data = f"{opname}:\n[execute] {execute_data}\n[setup] {setup_data}\n\n"
                    f.write(merged_data)


def clear_dump_task(args):
    if "onnx" in args.type and ("model" in args.type or "layer" in args.type):
        json_to_onnx(args)
    elif "cpu_profiling" in args.type:
        # 获取当前进程的cpu_profiling数据dump路径，新版CANN包需要加时间戳，否则不加时间戳
        cpu_profiling_path1 = os.path.join(os.environ.get(ATB_OUTPUT_DIR, ""), get_ait_dump_path(), "cpu_profiling")
        cpu_profiling_path2 = os.path.join(os.environ.get(ATB_OUTPUT_DIR, ""), "ait_dump", "cpu_profiling")
        if os.path.exists(cpu_profiling_path1):
            merge_cpu_profiling_data(cpu_profiling_path1)
        else:
            merge_cpu_profiling_data(cpu_profiling_path2)
    else:
        return
