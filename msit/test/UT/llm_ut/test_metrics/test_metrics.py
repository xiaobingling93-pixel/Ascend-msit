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
from unittest import TestCase

from msit_llm.metrics.metrics import Accuracy, EditDistance, get_metric


class TestMetrics(TestCase):

    def test_accuracy_score_positive(self):
        outs = ["1"]
        refs = ["1"]
        thr = 0

        self.accuracy = Accuracy(thr)
        generator = self.accuracy.compare_two_lists_of_words(outs, refs)

        with self.assertRaises(StopIteration):
            next(generator)

    def test_accuracy_score_negative(self):
        outs = ["1"]
        refs = [""]
        thr = 1
        
        self.accuracy = Accuracy(thr)
        generator = self.accuracy.compare_two_lists_of_words(outs, refs)

        self.assertEqual(next(generator), (0, 0))

    def test_edit_distance_positive(self):
        outs = ["a"]
        refs = ["a"]

        edit_distance = EditDistance(0)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)

        with self.assertRaises(StopIteration):
            next(generator)
    
    def test_edit_distance_diff_thr(self):
        outs = ["a"] * 1000000
        refs = [" "] * 1000000

        edit_distance = EditDistance(0)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)
        self.assertEqual(next(generator), (0, 1))

        edit_distance = EditDistance(2)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_get_metrics_should_raise_when_name_not_valid(self):
        with self.assertRaises(TypeError):
            get_metric(1)
        
        with self.assertRaises(TypeError):
            get_metric(1.5)
        
        with self.assertRaises(TypeError):
            get_metric([])

    def test_get_metrics_should_raise_when_name_not_support(self):
        with self.assertRaises(KeyError):
            get_metric("")

        with self.assertRaises(KeyError):
            get_metric("acc")        

    def test_get_metrics_should_raise_when_thr_not_valid(self):
        with self.assertRaises(TypeError):
            get_metric("accuracy", "")
