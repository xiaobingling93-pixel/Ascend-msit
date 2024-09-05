# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import json
import stat


def get_realpath_with_soft_link_check(file_path):
    if os.path.islink(os.path.abspath(file_path)):
        raise PermissionError('Opening softlink path is not permitted.')
    return os.path.realpath(file_path)


def check_file_owner_and_permissions(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File does not exist")
    real_file_path = get_realpath_with_soft_link_check(file_path)

    file_stat = os.stat(real_file_path)
    file_uid = file_stat.st_uid
    current_uid = os.getuid()

    if file_uid != current_uid:
        raise PermissionError("The file does not belong to the current user")
    if file_stat.st_mode & stat.S_IWOTH:
        raise PermissionError("Other users have write access")
    return real_file_path


def get_valid_read_file(file_path):
    real_file_path = check_file_owner_and_permissions(file_path)
    if not os.path.isfile(real_file_path):
        raise ValueError(f'Provided file_path={file_path} not exists or not a file.')
    if not os.access(real_file_path, os.R_OK):
        raise PermissionError(f'Opening file_path={file_path} is not permitted.')
    return real_file_path


def get_valid_read_directory(file_path):
    real_file_path = get_realpath_with_soft_link_check(file_path)
    if not os.path.isdir(real_file_path):
        raise ValueError(f'Provided file_path={file_path} not exists or not a directory.')
    if not os.access(real_file_path, os.R_OK | os.X_OK):
        raise PermissionError(f'Entering file_path={file_path} is not permitted.')
    return real_file_path


# 检查../../data/profiling目录中是否存在profiling文件，并检查该profiling文件是否正确配置，返回profiling文件路径
# profiling目录中只能存在一个profiling文件
def check_profiling_data(datapath):
    datapath = get_valid_read_directory(datapath)
    profiling_nums = 0
    for file in os.listdir(datapath):
        if file[0:5] == "PROF_":
            profiling_nums += 1
    if profiling_nums == 0:
        raise Exception(
            f"profiling data do not in {datapath},or the file name is incorrect."
            "Use the original name, such as PROF_xxxxx"
        )
    elif profiling_nums > 1:
        raise Exception("The number of profiling data is greater than 1, " "Please enter only one profiling data")
    datapath = os.path.join(datapath, os.listdir(datapath)[0])
    datapath = get_valid_read_directory(datapath)
    filename_not_correct = True
    for file in os.listdir(datapath):
        if file[0:7] == 'device_':
            filename_not_correct = False
    if filename_not_correct:
        raise ValueError(
            f'{datapath} is not a correct profiling file, \
                correct profiling file is PROF_xxxxxxxx and it includes device_*'
        )
    return datapath


def get_statistic_profile_data_path(profile_path):
    profile_path = get_valid_read_directory(profile_path)
    for device in os.listdir(profile_path):
        summary_path = os.path.join(profile_path, device, "summary")
        summary_path = get_valid_read_directory(summary_path)
        for file_name in os.listdir(summary_path):
            if "acl_statistic" in file_name:
                acl_statistic_path = os.path.join(summary_path, file_name)
    return get_valid_read_file(acl_statistic_path)
