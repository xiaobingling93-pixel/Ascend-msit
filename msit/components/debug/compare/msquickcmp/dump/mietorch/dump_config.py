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
import json
import os.path

from components.utils.file_open_check import ms_open
from msquickcmp.common import utils


class DumpConfig:
    def __init__(
        self,
        dump_path=".",
        mode='all',
        op_switch="off",
        api_list=None,
    ):
        dump_list_config = dict(model_name="Graph")
        if api_list:
            dump_list_config["layer"] = [api for api in api_list.split(",") if api]
        self.config = dict(
            dump=dict(
                dump_path=dump_path,
                dump_mode=mode,
                dump_op_switch=op_switch,
                dump_list=[dump_list_config]
            )
        )
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(cur_dir, "acl.json")
        try:
            with ms_open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)
        except FileNotFoundError:
            utils.logger.error("File not found.")
            raise
        except json.JSONDecodeError as e:
            utils.logger.error(f"JSON decode error:{e}")
            raise


