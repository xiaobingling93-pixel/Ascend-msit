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
import stat


class Const:

    ONLY_READ = stat.S_IRUSR | stat.S_IRGRP  # 0O440

    # bin path
    FAST_QUERY_BIN = os.path.join('tools', 'ms_fast_query', 'ms_fast_query.py')

    # error detail
    ERR_UNSUPPORT = 'Op is unsupported.'
    ERR_OPP_NOT_EXIST = 'Opp not exist.'
