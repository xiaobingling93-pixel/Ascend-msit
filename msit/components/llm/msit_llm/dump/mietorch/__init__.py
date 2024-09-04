# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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


mindie_rt_dump_so_path = "tools/ait_backend/mindie_torch_dump/libmindiedump.so"
if os.environ.get("ASCEND_TOOLKIT_HOME", None) is None:
    raise EnvironmentError("Please set ASCEND_TOOLKIT_HOME by running set_env.sh.")
mindie_rt_dump_so_path = os.path.join(os.environ["ASCEND_TOOLKIT_HOME"], mindie_rt_dump_so_path)

cur_dir = os.path.dirname(os.path.abspath(__file__))
mindie_rt_dump_config = os.path.join(cur_dir, "acl.json")
os.environ["LD_PRELOAD"] = f'{mindie_rt_dump_so_path}:{os.environ.get("LD_PRELOAD", "")}'
os.environ["MINDIE_RT_DUMP_CONFIG_PATH"] = mindie_rt_dump_config


