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
from unittest.mock import patch, MagicMock

from auto_optimizer import KnowledgeFactory
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer, InferTestConfig


class TestOptimize(unittest.TestCase):
    def setUp(self):
            self.knowledges = KnowledgeFactory.get_knowledge_pool()
            self.optimizer = GraphOptimizer(list(self.knowledges.keys()))
            self.graph = MagicMock(spec=BaseGraph)

    @patch("auto_optimizer.graph_refactor.interface.base_graph.BaseGraph")
    @patch("auto_optimizer.pattern.knowledges.knowledge_base.KnowledgeBase")
    def test_optimize_with_valid_inputs(self, mock_knowledge_base, mock_base_graph):
        """测试 optimize 函数在正常情况下的行为"""
        # 设置 mock 对象的行为
        mock_knowledge_base.pre_process.return_value = True
        mock_knowledge_base.has_next_pattern.side_effect = [True, False]
        mock_knowledge_base.next_pattern.return_value = None
        mock_knowledge_base.match_pattern.return_value = ['match1', 'match2']
        mock_knowledge_base.has_next_apply.side_effect = [True, False]
        mock_knowledge_base.next_apply.return_value = None
        mock_knowledge_base.apply.side_effect = [True, False]
        mock_knowledge_base.post_process.return_value = True

        # 调用被测试的函数
        result = GraphOptimizer.optimize(mock_base_graph, mock_knowledge_base)

        # 验证结果
        self.assertTrue(result)
        mock_knowledge_base.pre_process.assert_called_once_with(mock_base_graph)
        mock_knowledge_base.has_next_pattern.assert_called()
        mock_knowledge_base.next_pattern.assert_called_once()
        mock_knowledge_base.match_pattern.assert_called_once_with(mock_base_graph)
        mock_knowledge_base.has_next_apply.assert_called()
        mock_knowledge_base.next_apply.assert_called_once()
        mock_knowledge_base.apply.assert_any_call(mock_base_graph, 'match1')
        mock_knowledge_base.apply.assert_any_call(mock_base_graph, 'match2')
        mock_knowledge_base.post_process.assert_called_once_with(mock_base_graph)

    @patch("auto_optimizer.graph_refactor.interface.base_graph.BaseGraph")
    @patch("auto_optimizer.pattern.knowledges.knowledge_base.KnowledgeBase")
    def test_optimize_with_pre_process_failure(self, mock_knowledge_base, mock_base_graph):
        """测试 optimize 函数在 pre_process 失败时的行为"""
        # 设置 mock 对象的行为
        mock_knowledge_base.pre_process.return_value = False

        # 调用被测试的函数
        result = GraphOptimizer.optimize(mock_base_graph, mock_knowledge_base)

        # 验证结果
        self.assertFalse(result)
        mock_knowledge_base.pre_process.assert_called_once_with(mock_base_graph)
        mock_knowledge_base.has_next_pattern.assert_not_called()
        mock_knowledge_base.next_pattern.assert_not_called()
        mock_knowledge_base.match_pattern.assert_not_called()
        mock_knowledge_base.has_next_apply.assert_not_called()
        mock_knowledge_base.next_apply.assert_not_called()
        mock_knowledge_base.apply.assert_not_called()
        mock_knowledge_base.post_process.assert_not_called()

    @patch("auto_optimizer.graph_refactor.interface.base_graph.BaseGraph")
    @patch("auto_optimizer.pattern.knowledges.knowledge_base.KnowledgeBase")
    def test_optimize_with_no_match_results(self, mock_knowledge_base, mock_base_graph):
        """测试 optimize 函数在没有匹配结果时的行为"""
        # 设置 mock 对象的行为
        mock_knowledge_base.pre_process.return_value = True
        mock_knowledge_base.has_next_pattern.side_effect = [True, False]
        mock_knowledge_base.next_pattern.return_value = None
        mock_knowledge_base.match_pattern.return_value = []
        mock_knowledge_base.has_next_apply.side_effect = [True, False]
        mock_knowledge_base.next_apply.return_value = None
        mock_knowledge_base.apply.return_value = False
        mock_knowledge_base.post_process.return_value = True

        # 调用被测试的函数
        result = GraphOptimizer.optimize(mock_base_graph, mock_knowledge_base)

        # 验证结果
        self.assertFalse(result)
        mock_knowledge_base.pre_process.assert_called_once_with(mock_base_graph)
        mock_knowledge_base.has_next_pattern.assert_called()
        mock_knowledge_base.next_pattern.assert_called_once()
        mock_knowledge_base.match_pattern.assert_called_once_with(mock_base_graph)
        mock_knowledge_base.has_next_apply.assert_not_called()
        mock_knowledge_base.next_apply.assert_not_called()
        mock_knowledge_base.apply.assert_not_called()
        mock_knowledge_base.post_process.assert_called_once_with(mock_base_graph)
        
    @patch('auto_optimizer.graph_optimizer.GraphOptimizer.optimize')
    def test_apply_knowledges_with_graph(self, mock_optimize):
        mock_optimize.return_value = True
        graph, applied_knowledges = self.optimizer.apply_knowledges(self.graph)
        self.assertIsInstance(graph, BaseGraph)
        self.assertIsInstance(applied_knowledges, list)
        self.assertTrue(mock_optimize.called)

    @patch('auto_optimizer.graph_optimizer.GraphOptimizer.optimize')
    @patch('auto_optimizer.graph_optimizer.GraphOptimizer._effective')
    @patch('auto_optimizer.inference_engine.model_convert.onnx2om')
    @patch('tempfile.gettempdir')
    @patch('os.getpid')
    @patch('os.rename')
    @patch('os.remove')
    @patch('multiprocessing.get_context')
    def test_apply_knowledges_with_infer_test(
        self,
        mock_get_context,
        mock_remove,
        mock_rename,
        mock_getpid,
        mock_gettempdir,
        mock_onnx2om,
        mock_effective,
        mock_optimize
    ):
        mock_optimize.return_value = True
        mock_effective.return_value = True
        mock_gettempdir.return_value = '/tmp'
        mock_getpid.return_value = 1234
        mock_onnx2om.return_value = '/tmp/model.om'
        mock_get_context.return_value = MagicMock()
        mock_get_context.return_value.Queue.return_value = MagicMock()
        mock_get_context.return_value.Queue.return_value.qsize.return_value = 3
        mock_get_context.return_value.Queue.return_value.get.return_value = True
        mock_get_context.return_value.Process.return_value.start = MagicMock()
        mock_get_context.return_value.Process.return_value.join = MagicMock()

        cfg = InferTestConfig()
        graph, applied_knowledges = self.optimizer.apply_knowledges_with_infer_test(self.graph, cfg)

        self.assertIsInstance(graph, BaseGraph)
        self.assertIsInstance(applied_knowledges, list)
        self.assertTrue(mock_optimize.called)
        self.assertTrue(mock_effective.called)
        self.assertTrue(mock_onnx2om.called)
        self.assertTrue(mock_gettempdir.called)
        self.assertTrue(mock_getpid.called)
        self.assertTrue(mock_rename.called)
        self.assertTrue(mock_remove.called)
        self.assertTrue(mock_get_context.called)
        
if __name__ == '__main__':
    unittest.main()