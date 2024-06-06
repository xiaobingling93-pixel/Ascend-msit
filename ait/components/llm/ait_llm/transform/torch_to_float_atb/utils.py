# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#cd 
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


def init_save_name(save_name):
    if os.path.splitext(save_name)[-1] in [".c", ".cpp", ".h", ".hpp"]:
        save_name = os.path.splitext(save_name)[0]
    return os.path.basename(save_name)


def init_save_dir(save_dir, sub_dir):
    save_dir = os.path.abspath(save_dir)
    if os.path.basename(save_dir) in ["model", "layer"]:
        save_dir = os.path.dirname(save_dir)
    save_dir = os.path.join(save_dir, sub_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir