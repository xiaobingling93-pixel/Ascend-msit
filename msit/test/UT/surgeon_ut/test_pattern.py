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
import unittest

from auto_optimizer.pattern.pattern import MatchPattern
from auto_optimizer.pattern.pattern import Pattern


class TestPattern(unittest.TestCase):

    def test_add_node_func_0(self):
        pattern = Pattern()
        pattern.add_node('Conv_0', ['Conv'], None)
        pattern.add_node('Conv_1', ['Conv'], None)

        self.assertEqual(len(pattern.node_dict), 2)
        self.assertNotEqual(pattern.node_dict.get('Conv_0'), None)
        self.assertNotEqual(pattern.node_dict.get('Conv_1'), None)

    def test_add_node_func_1(self):
        pattern = Pattern()

        try:
            pattern.add_node('Conv', ['Conv'], None)
        except RuntimeError as e:
            pass
        try:
            pattern.add_node('Conv', ['Conv'], None)
        except RuntimeError as e:
            pass

        self.assertEqual(len(pattern.node_dict), 1)
        self.assertNotEqual(pattern.node_dict.get('Conv'), None)

    def test_node_can_match_more_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_node_loop('Conv', MatchPattern.MATCH_ONCE_OR_MORE) \
            .set_node_loop('Relu', MatchPattern.MATCH_ZERO_OR_MORE)

        self.assertTrue(pattern.node_dict['Conv'].can_match_more_time())
        self.assertTrue(pattern.node_dict['Relu'].can_match_more_time())

    def test_node_can_match_zero_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_node_loop('Conv', MatchPattern.MATCH_ONCE_OR_MORE) \
            .set_node_loop('Relu', MatchPattern.MATCH_ZERO_OR_MORE)

        self.assertFalse(pattern.node_dict['Conv'].can_match_zero_time())
        self.assertTrue(pattern.node_dict['Relu'].can_match_zero_time())

    def test_cann_match_more_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_loop(MatchPattern.MATCH_ONCE_OR_MORE)

        self.assertTrue(pattern.can_match_more())


if __name__ == "__main__":
    unittest.main()
