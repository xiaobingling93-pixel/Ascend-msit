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
import subprocess

from components.utils.util import filter_cmd
from components.utils.security_check import check_input_path_legality


def get_msaccucmp_path():
    # 加载CANN TOOLKIT环境变量，该变量在分包安装下不存在，需要手动进行配置
    cann_path = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")
    msaccucmp_path = os.path.join(cann_path, "tools", "operator_cmp", "compare", "msaccucmp.py")
    msaccucmp_path = check_input_path_legality(msaccucmp_path)
    return msaccucmp_path


def execute_convert_npy_command(command):
    try:
        command = filter_cmd(command)
        result = subprocess.run(command, shell=False, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error while converting bin data to npy: {result.stderr.strip()}"
    except Exception as e:
        return f"Error while converting bin data to npy: {e}"
    

def convert_ge_dump_file_to_npy(input_path, npy_path):
    # input_path可以指定单个文件或者是目录
    msaccucmp_path = get_msaccucmp_path()
    command = ["python3", msaccucmp_path, "convert", "-d", input_path, "-out", npy_path]
    return_msg = execute_convert_npy_command(command)
    if "Error while converting bin data to npy:" in return_msg:
        raise RuntimeError("Execute msaccucmp convert failed, Please check CANN Environment.")
    

def get_ascbackend_ascgraph(ascgen_path):
    ascgraph_list = []
    for element in os.listdir(ascgen_path):
        # 当前不支持对fusedgraph进行解析构图
        if element.endswith(".py") and not element.startswith("autofuse_fused"):
            ascgraph_list.append(element)
    return ascgraph_list