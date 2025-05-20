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

import unittest
from unittest.mock import patch

from components.llm.msit_llm.compare.cmp_mgr import CompareMgr


class TestFilterTensorPaths(unittest.TestCase):
    # 基础测试数据
    QKV_PATHS = [
        "0_Attention/0_QKVLinearSplitPack/after/outtensor0.bin",
        "0_Attention/0_QKVLinearSplitPack/after/outtensor1.bin",
        "0_Attention/0_QKVLinearSplitPack/after/outtensor2.bin",
        "0_Attention/1_RotaryPositionEmbedding/before/intensor0.bin"
    ]
    
    QKV_SPLIT_PATHS = [
        "0_Attention/0_QKVLinearSplitPack/1_SplitOperation/after/outtensor0.bin",
        "0_Attention/0_QKVLinearSplitPack/1_SplitOperation/after/outtensor1.bin",
        "0_Attention/0_QKVLinearSplitPack/1_SplitOperation/after/outtensor2.bin"
    ]

    def test_qkv_mode(self):
        """验证标准qkv模式的路径过滤逻辑"""
        test_cases = [
            # (golden_paths, my_paths, 预期结果)
            (
                ["root.model.layers.0.self_attn.q_proj/output.pth"],
                [self.QKV_PATHS[0], "other.bin"],
                [self.QKV_PATHS[0]]
            ),
            (
                ["root.model.layers.5.self_attn.k_proj/output.pth"],
                [self.QKV_PATHS[1], "unrelated.bin"],
                [self.QKV_PATHS[1]]
            ),
            (
                [
                    "root.model.layers.9.self_attn.v_proj/output.pth",
                    "root.model.layers.9.self_attn.q_proj/output.pth"
                ],
                [self.QKV_PATHS[2], self.QKV_PATHS[3]],
                [self.QKV_PATHS[2], self.QKV_PATHS[3]] 
            ),
            (
                ["invalid_path.pth"],
                self.QKV_PATHS,
                self.QKV_PATHS 
            )
        ]
        
        for golden, my, expected in test_cases:
            with self.subTest(golden=golden, my=my):
                result = CompareMgr.filter_tensor_paths(golden, my, 'qkv')
                self.assertCountEqual(result, expected)  # 顺序无关的集合比较


    def test_qkv_split_mode(self):
        """验证split模式的特殊处理逻辑"""
        test_cases = [
            (
                ["root.model.layers.3.self_attn.q_proj/output.pth"],
                [self.QKV_SPLIT_PATHS[0], "noise.bin"],
                [self.QKV_SPLIT_PATHS[0]]
            ),
            (
                [
                    "root.model.layers.0.self_attn.k_proj/output.pth",
                    "root.model.layers.0.self_attn.v_proj/output.pth"
                ],
                self.QKV_SPLIT_PATHS,
                self.QKV_SPLIT_PATHS[1:] 
            ),
            (
                [],
                ["any_path.bin"],
                ["any_path.bin"] 
            )
        ]
        
        for golden, my, expected in test_cases:
            with self.subTest(golden=golden, my=my):
                result = CompareMgr.filter_tensor_paths(golden, my, 'qkv_split')
                self.assertCountEqual(result, expected)

    def test_invalid_path_prefix(self):
        """验证非法path_prefix的异常抛出"""
        with self.assertRaisesRegex(ValueError, "Unsupported path_prefix: invalid"):
            CompareMgr.filter_tensor_paths([], [], 'invalid')