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

import json
import os
import stat
import pickle

TIMEOUT = 60 * 60 * 24
OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR

READ_FLAGS = os.O_RDONLY


class IOUtil:
    """
    Define common utils
    """

    @staticmethod
    def remove_subdirectory(directories):
        """移除子目录"""
        if len(directories) == 0:
            return []
        flag = [0] * len(directories)
        for item1, item1_value in enumerate(directories[:-1]):
            if flag[item1]:
                continue
            for item2, item2_value in enumerate(directories[item1 + 1:],
                                                start=item1 + 1):
                if flag[item2]:
                    continue
                if os.path.commonprefix([item1_value, item2_value]) == \
                        item1_value:
                    flag[item2] = 1
                if os.path.commonprefix([item1_value, item2_value]) == \
                        item2_value:
                    flag[item1] = 1
        return [item_value for item, item_value in enumerate(directories) if not flag[item]]

    @staticmethod
    def check_path_is_empty(real_path):
        """检查文件夹是否为空文件夹"""
        if os.path.isdir(real_path):
            for _, _, files in os.walk(real_path):
                if files:
                    return False
            return True
        return not os.path.isfile(real_path)

    @staticmethod
    def file_safe_write(obj, file):
        with os.fdopen(os.open(file, OPEN_FLAGS, OPEN_MODES), 'w') as fout:
            fout.write(obj)

    @staticmethod
    def json_safe_dump(obj, file):
        with os.fdopen(os.open(file, OPEN_FLAGS, OPEN_MODES), 'w') as fout:
            json.dump(obj, fout, indent=4, ensure_ascii=False)

    @staticmethod
    def json_safe_load(file):
        if not os.path.exists(file):
            raise Exception(f'File {file} is not existed!')

        with os.fdopen(os.open(file, READ_FLAGS, OPEN_MODES), 'r') as fout:
            data = json.load(fout)
        return data
