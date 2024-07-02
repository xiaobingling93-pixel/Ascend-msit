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
import platform

MAX_JSON_FILE_SIZE = 10 * 1024 ** 2
WINDOWS_PATH_LENGTH_LIMIT = 200
LINUX_FILE_NAME_LENGTH_LIMIT = 200


class SoftLinkCheckException(Exception):
    pass


def islink(path):
    path = os.path.abspath(path)
    return os.path.islink(path)


def check_path_length_valid(path):
    path = os.path.realpath(path)
    if platform.system().lower() == 'windows':
        return len(path) <= WINDOWS_PATH_LENGTH_LIMIT
    else:
        return len(os.path.basename(path)) <= LINUX_FILE_NAME_LENGTH_LIMIT


def check_input_file_valid(input_path, max_file_size=MAX_JSON_FILE_SIZE):
    if islink(input_path):
        raise SoftLinkCheckException("Input path doesn't support soft link.")

    input_path = os.path.realpath(input_path)
    if not os.path.exists(input_path):
        raise ValueError('Input file %s does not exist!' % input_path)

    if not os.access(input_path, os.R_OK):
        raise PermissionError('Input file %s is not readable!' % input_path)

    if not check_path_length_valid(input_path):
        raise ValueError('The real path or file name of input is too long.')

    if os.path.getsize(input_path) > max_file_size:
        raise ValueError(f'The file is too large, exceeds {max_file_size // 1024 ** 2}MB')
