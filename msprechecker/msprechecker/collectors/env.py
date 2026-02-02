# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os

from .base import BaseCollector


class EnvCollector(BaseCollector):
    ENV_FILTERS = [
        "ASCEND", "MINDIE", "ATB_", "HCCL_", "MIES", 
        "RANKTABLE", "GE_", "TORCH", "ACL_", "NPU_",
        "LCCL_", "LCAL_", "OPS", "INF_"
    ]

    def __init__(self, error_handler=None, *, filter_env: bool = False):
        super().__init__(error_handler)
        self.error_handler.type = "env"
        self.filter_env = filter_env

    def _collect_data(self):
        try:
            env_items = os.environ.items()
        except Exception as e:
            self.error_handler.add_error(
                filename=__file__,
                function="_collect_data",
                lineno=35,
                what="使用 'os.environ' 获取环境信息失败",
                reason=str(e)
            )
            env_items = {}

        if self.filter_env:
            return {
                k: v for k, v in env_items
                if any(item in k for item in self.ENV_FILTERS)
            }
        return dict(env_items)
