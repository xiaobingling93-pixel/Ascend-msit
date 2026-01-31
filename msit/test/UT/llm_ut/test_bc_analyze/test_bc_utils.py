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
import pytest
import os
from unittest.mock import patch, MagicMock
import random
from msit_llm.bc_analyze import RandomNameSequence

class TestRandomNameSequence:
    
    def test_character_set(self):
        assert set(RandomNameSequence.characters) == set("abcdefghijklmnopqrstuvwxyz0123456789_")
    
    def test_name_generation(self):
        """测试生成随机名称的基本功能"""
        rns = RandomNameSequence()
        names = [next(rns) for _ in range(10)]
        
        for name in names:
            assert len(name) == 8
            assert all(c in RandomNameSequence.characters for c in name)