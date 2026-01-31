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
from random import Random
from datetime import datetime, timedelta, timezone


class RandomNameSequence(object):
    characters = "abcdefghijklmnopqrstuvwxyz0123456789_"

    def __init__(self):
        self._rng = None
        self._rng_pid = None

    def __iter__(self):
        return self
    
    def __next__(self):
        return ''.join(self.rng.choices(self.characters, k=8))

    @property
    def rng(self):
        cur_pid = os.getpid()
        if cur_pid != getattr(self, '_rng_pid', None):
            self._rng = Random()
            self._rng_pid = cur_pid
        return self._rng


def get_timestamp():
    cst_timezone = timezone(timedelta(hours=8))
    current_time = datetime.now(cst_timezone)
    return current_time.strftime("%Y%m%d%H%M%S")
