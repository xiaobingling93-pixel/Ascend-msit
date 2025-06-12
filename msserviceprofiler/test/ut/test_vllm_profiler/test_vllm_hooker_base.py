# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
import sys
import traceback
import unittest
from unittest.mock import MagicMock, patch
from packaging.version import Version
from msserviceprofiler.vllm_profiler.vllm_profiler_core.vllm_hooker_base import VLLMHookerBase


class TestVLLMHookerBase(unittest.TestCase):
    def setUp(self):
        # 创建测试用的子类
        class TestHooker(VLLMHookerBase):
            pass
        
        self.hooker = TestHooker()
    
    def test_support_version_no_constraint(self):
        """测试没有版本约束时总是返回 True"""
        self.assertTrue(self.hooker.support_version("1.0.0"))
        self.assertTrue(self.hooker.support_version("2.0.0"))
    
    def test_support_version_min_only(self):
        """测试只有最小版本约束的情况"""
        self.hooker.vllm_version = ("1.5.0", None)
        
        # 低于最小版本
        self.assertFalse(self.hooker.support_version("1.4.9"))
        # 等于最小版本
        self.assertTrue(self.hooker.support_version("1.5.0"))
        # 高于最小版本
        self.assertTrue(self.hooker.support_version("1.5.1"))
        self.assertTrue(self.hooker.support_version("2.0.0"))
    
    def test_support_version_max_only(self):
        """测试只有最大版本约束的情况"""
        self.hooker.vllm_version = (None, "1.8.0")
        
        # 低于最大版本
        self.assertTrue(self.hooker.support_version("1.7.9"))
        # 等于最大版本
        self.assertTrue(self.hooker.support_version("1.8.0"))
        # 高于最大版本
        self.assertFalse(self.hooker.support_version("1.8.1"))
        self.assertFalse(self.hooker.support_version("2.0.0"))
    
    def test_support_version_range(self):
        """测试版本范围约束"""
        self.hooker.vllm_version = ("1.6.0", "1.9.0")
        
        # 边界检查
        self.assertFalse(self.hooker.support_version("1.5.9"))
        self.assertTrue(self.hooker.support_version("1.6.0"))
        self.assertTrue(self.hooker.support_version("1.7.5"))
        self.assertTrue(self.hooker.support_version("1.9.0"))
        self.assertFalse(self.hooker.support_version("1.9.1"))
    
    def test_support_version_invalid_input(self):
        """测试无效版本号处理"""
        self.hooker.vllm_version = ("1.0.0", "2.0.0")
        
        with self.assertRaises(Exception):
            # 无效版本号应引发异常
            self.hooker.support_version("invalid_version")
    
    def test_get_parents_name_direct_call(self):
        """测试直接调用栈分析"""
        def call_get_parents_name():
            # 调用目标方法
            return VLLMHookerBase.get_parents_name(None)
        
        # 获取当前函数名
        current_func = call_get_parents_name.__name__
        
        # 调用并验证
        parent_name = call_get_parents_name()
        self.assertEqual(parent_name, "test_get_parents_name_direct_call")
    
    def test_get_parents_name_nested_call(self):
        """测试嵌套调用栈分析"""
        def level3():
            return VLLMHookerBase.get_parents_name(None, index=3)
        
        def level2():
            return level3()
        
        def level1():
            return level2()
        
        # 获取当前函数名
        current_test_func = "test_get_parents_name_nested_call"
        
        # 验证不同索引值
        self.assertEqual(level1(), current_test_func)  # index=3
    
    def test_get_parents_name_out_of_range(self):
        """测试超出调用栈范围的情况"""
        # 在顶层直接调用
        result = VLLMHookerBase.get_parents_name(None, index=100)
        self.assertIsNone(result)
    
    def test_get_parents_name_with_real_function(self):
        """测试真实函数作为参数的场景"""
        def target_function():
            # 空函数，仅用于测试
            pass
        
        def caller_function():
            return VLLMHookerBase.get_parents_name(target_function)
        
        # 调用并验证
        parent_name = caller_function()
        self.assertEqual(parent_name, "test_get_parents_name_with_real_function")
    
    def test_get_parents_name_index_zero(self):
        """测试索引为0的特殊情况"""
        def call_with_index_zero():
            return VLLMHookerBase.get_parents_name(None, index=0)
        
        # 索引0应该返回get_parents_name的直接调用者
        parent_name = call_with_index_zero()
        self.assertEqual(parent_name, "call_with_index_zero")
