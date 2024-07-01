# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import logging
import subprocess
import sys
import os
import pytest
import onnx

from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor import Node
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.accuracy_locat import accuracy_locat as al

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    # staticmethod or classmethod
    @classmethod
    def get_base_path(cls):
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        return _current_dir

    @classmethod
    def get_cann_path(cls):
        result = subprocess.run(['which', 'atc'], stdout=subprocess.PIPE)
        atc_path = result.stdout.decode('utf-8').strip()
        cann_path = atc_path[:-8]
        return cann_path

    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    def init(self):
        self.cann_path = self.get_cann_path()
        self.args = CmpArgsAdapter(
            gold_model=os.path.join(self.get_base_path(), 'onnx/resnet18_static.onnx'),
            om_model=os.path.join(self.get_base_path(), 'om/resnet18_static.om'),
            weight_path="",
            input_data_path="",
            cann_path=self.cann_path,
            out_path=os.path.join(self.get_base_path(), '/test/resnet18/output/'),
            input_shape="",
            device="0",
            output_size="",
            output_nodes="",
            advisor=False,
            dym_shape_range="",
            dump=True,
            bin2npy=False,
            custom_op="",
            locat=True,
        )

    def test_calculate_flow(self):
        model = onnx.load(self.args.model_path)
        graph = model.graph

        startnode_name = graph.node[0].name
        endnode_name = graph.node[-1].name

        og = OnnxGraph.parse(self.args.model_path)
        startnode: Node = og.get_node(startnode_name, node_type=Node)
        endnode: Node = og.get_node(endnode_name, node_type=Node)
        satisfied_nodes = []
        satisfied_nodes = al.calculate_flow(og, startnode, endnode)
        linear_size_real = 22
        linear_size = len(satisfied_nodes)
        assert linear_size == linear_size_real
