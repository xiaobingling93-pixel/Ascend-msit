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
import datetime
import pytest
from msquickcmp.cmp_process import cmp_process
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    '''
    just for smoking test， not for pipeline or DT
    '''

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
    def set_accumulate_cmp_args(cls):
        args_data2vec_cmp = CmpArgsAdapter(
            gold_model=os.path.join(cls.get_base_path(), 'onnx/data2vec_1_108.onnx'),
            om_model=os.path.join(cls.get_base_path(), 'om/data2vec_1_108.om'),
            weight_path="",
            input_data_path=os.path.realpath(
                "{},{}".format(
                    os.path.join(cls.get_base_path(), 'input_datas/data2vec/1535_0.bin'),
                    os.path.join(cls.get_base_path(), 'input_datas/data2vec/1535_1.bin'),
                )
            ),
            cann_path=cls.cann_path,
            out_path=os.path.join(cls.get_base_path(), 'output/'),
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
        return args_data2vec_cmp

    @classmethod
    def set_single_node_cmp_args(cls):
        args_gelu_cmp = CmpArgsAdapter(
            gold_model=os.path.join(cls.get_base_path(), 'onnx/695_703.onnx'),
            om_model=os.path.join(cls.get_base_path(), 'om/695_703.om'),
            weight_path="",
            input_data_path=os.path.join(cls.get_base_path(), 'input_datas/gelu/695.bin'),
            cann_path=cls.cann_path,
            out_path=os.path.join(cls.get_base_path(), 'output/'),
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
        return args_gelu_cmp

    @classmethod
    def get_latest_dir(cls):
        cur_path = os.path.join(cls.get_base_path(), "output/")
        latest_timestamp = 0
        latest_dir_path = ""

        for item in os.listdir(cur_path):
            item_path = os.path.join(cur_path, item)
            if os.path.isdir(item_path):
                timestamp = os.path.getmtime(item_path)
                dt = datetime.datetime.fromtimestamp(timestamp)
                if timestamp > latest_timestamp:
                    latest_dir_path = item_path
        return latest_dir_path

    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    def init(self):
        self.cann_path = self.get_cann_path()

        self.args_data2vec_cmp = self.set_accumulate_cmp_args()
        self.args_gelu_cmp = self.set_single_node_cmp_args()

    # =======================testcases=============================

    def test_compare_accumlate_accuracy_area_situation(self):
        '''
        存在累计误差区间的场景
        '''

        cmp_process(self.args_data2vec_cmp, True)
        latest_path = self.get_latest_dir()
        log_path = os.path.join(latest_path, "error_interval_info.txt")
        assert os.path.exists(log_path)

    def test_compare_cause_by_single_node_situation(self):
        '''
        正常的场景
        '''
        cmp_process(self.args_gelu_cmp, True)
        latest_path = self.get_latest_dir()
        log_path = os.path.join(latest_path, "error_interval_info.txt")
        assert not os.path.exists(log_path)
