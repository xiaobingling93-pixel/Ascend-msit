# -*- coding: utf-8 -*-
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

from components.debug.compare.msquickcmp.common.args_check import (
    safe_string, check_cann_path_legality
)

ascend_toolkit_home_path = os.getenv("ASCEND_TOOLKIT_HOME")
if not ascend_toolkit_home_path:
    raise EnvironmentError("Please first source CANN environment by running set_env.sh.")
ascend_toolkit_home_path = safe_string(ascend_toolkit_home_path)
ascend_toolkit_home_path = check_cann_path_legality(ascend_toolkit_home_path)  # check cann path

path_components = ["tools", "ait_backend", "mindie_torch_dump", "libmindiedump.so"]
mindie_rt_dump_so_path = os.path.join(ascend_toolkit_home_path, *path_components)

cur_dir = os.path.dirname(os.path.abspath(__file__))
mindie_rt_dump_config = os.path.join(cur_dir, "acl.json")
ld_preload = os.getenv("LD_PRELOAD")
if ld_preload:
    os.environ["LD_PRELOAD"] = f'{mindie_rt_dump_so_path}:{ld_preload}'
else:
    os.environ["LD_PRELOAD"] = mindie_rt_dump_so_path

os.environ["MINDIE_RT_DUMP_CONFIG_PATH"] = mindie_rt_dump_config
