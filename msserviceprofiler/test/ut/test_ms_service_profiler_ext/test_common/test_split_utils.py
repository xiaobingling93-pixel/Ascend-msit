# -*- coding: utf-8 -*-
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
import pandas as pd
import numpy as np

from msserviceprofiler.ms_service_profiler_ext.common.split_utils import (
    preprocess_framework_df,
    postprocess_framework_df,
    get_batch_framework,
    get_filter_rule_df,
)

class TestUtilsFuctions(unittest.TestCase):
    def test_preprocess_framework_df_with_no_name(self):
        data = {
            'pid': ['40'],
        }
        framework_df = pd.DataFrame(data)

        result = preprocess_framework_df(framework_df)
        self.assertIsNone(result)

    
    def test_postprocess_framework_df_null(self):
        # 创建测试数据
        framework_df = pd.DataFrame()
        post_event_pairs = []
        name = 'Prefill'

        # 调用你的函数
        result = postprocess_framework_df(framework_df, post_event_pairs, name)

        # 验证结果
        self.assertTrue(result.empty)
    
    def test_get_batch_framework_sample(self):
        framework_df = pd.DataFrame({
            'name': ['forward', 'prepareInputs'],
            'start_time(microsecond)': ['100', '200'],
            'during_time(microsecond)': ['2000', '3000'],
            'pid': [40, 40],
            'tid': [100, 100]
        })
        name = 'Decode'

        result = get_batch_framework(framework_df, name)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result['name'].values.tolist(), ['forward', 'prepareInputs'])

        framework_df = pd.DataFrame({
            'name': ['other'],
            'start_time(microsecond)': ['100'],
            'during_time(microsecond)': ['3000'],
            'pid': [40],
            'tid': [100]
        })
        name = 'Decode'

        result = get_batch_framework(framework_df, name)
        self.assertTrue(result.empty)

    def test_get_filter_rule_df(self):
        # 创建测试数据
        framework_df = pd.DataFrame({
            'name': ['encode', 'processPythonExecResult', 'serializeExcueteMessage', 
                     'other', 'serializeExcueteMessage', 'httpRes'],
            'during_time(microsecond)': [1000, 2000, 3000, 4000, 5000, 6000],
            'pid': [40, 40, 40, 40, 40, 40],
            'tid': [100, 100, 100, 100, 100, 100]
        })
        result = get_filter_rule_df(framework_df)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(
            result['name'].values.tolist(), 
            ['serializeExcueteMessage', 'other', 'serializeExcueteMessage'])