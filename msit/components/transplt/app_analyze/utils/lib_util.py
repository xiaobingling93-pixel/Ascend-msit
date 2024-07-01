# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
from app_analyze.common.kit_config import KitConfig


def is_acc_path(file):
    acc_paths = KitConfig.INCLUDES.values()
    return any(file.startswith(p) for p in acc_paths)


def get_sys_path():
    sys_paths = ['/usr/']

    _cmake_prefix_path(sys_paths)
    _opencv_dir_path(sys_paths)
    _cuda_path(sys_paths)
    _opencv_include_path(sys_paths)

    return sys_paths


def _cmake_prefix_path(sys_paths):
    cmake_prefix_path = os.environ.get('CMAKE_PREFIX_PATH')
    if cmake_prefix_path:
        cmake_prefix_path = filter(bool, cmake_prefix_path.split(':'))  # 去空字符穿
        for p in cmake_prefix_path:
            if not p.startswith('/usr/'):
                sys_paths.append(p)


def _opencv_dir_path(sys_paths):
    # The directory containing a CMake configuration file for OpenCV.
    opencv_dir = os.environ.get('OpenCV_DIR')
    if opencv_dir and not opencv_dir.startswith('/usr/'):
        opencv_dir = os.path.normpath(opencv_dir + '/../../../')
        sys_paths.append(opencv_dir)


def _cuda_path(sys_paths):
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if not cuda_home.startswith('/usr/'):
        sys_paths.append(cuda_home)


def _opencv_include_path(sys_paths):
    for include in KitConfig.INCLUDES.values():
        if include and not include.startswith('/usr/'):
            sys_paths.append(include)
