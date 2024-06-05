# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import itertools
from ait_llm.common.log import logger
from ait_llm.compare.cmp_op_match import OpMatchMgr
from ait_llm.compare.cmp_data_parse import CompareDataParse, CompareDataTorch, CompareDataATB
from ait_llm.compare.cmp_utils import BasicDataInfo, fill_row_data, save_compare_reault_to_csv
from ait_llm.dump.torch_dump.topo import ModelTree, TreeNode


class CompareMgr:
    data_parsers = [CompareDataATB, CompareDataTorch]

    def __init__(self, golden_path: str, my_path: str, args) -> None:
        self.args = args
        self.golden_data: CompareDataParse = self.init_compare_data(golden_path, args)
        self.my_data: CompareDataParse = self.init_compare_data(my_path, args)
        self.op_match: OpMatchMgr = OpMatchMgr(args)
        self.compared_result = []  # 收集结果

    def init_compare_data(self, data_path: str, args) -> CompareDataParse:
        for cls_data in self.data_parsers:
            if cls_data.accept(data_path, args):
                return cls_data(data_path, self.args)
        logger.error(f"cannot parse data path({data_path}). it is not a atb dump path or a torch dump path.")
        return None

    def is_parsed_cmp_path(self):
        return self.golden_data is not None and self.my_data is not None

    def compare(self, output_path="."):
        self.compared_result = []

        op_map = self.op_match.match(self.golden_data, self.my_data)
        # 同步校验tokenid
        golden_tokens = self.golden_data.getCmpToken()
        my_tokens = self.my_data.getCmpToken()
        golden_token_set = set(self._flatten_and_enum_tuple(golden_tokens))
        my_token_set = set(self._flatten_and_enum_tuple(my_token_set))
        if len(golden_tokens) == 1 and len(my_tokens) == 1:
            # 指定token，强制比对
            self.compare_token(golden_tokens, my_tokens, op_map)
        elif not golden_token_set.isdisjoint(my_token_set):
            logger.error(f"my tokens is {my_tokens} and golden tokens is {golden_tokens}. The two cannot be matched.")
            return None
        else:
            cmp_tokens = golden_token_set.intersection(my_token_set)
            for token_id in cmp_tokens:
                self.compare_token(token_id, token_id, op_map)

        return save_compare_reault_to_csv(self.compared_result, output_path)

    def compare_token(self, golden_token_id, my_token_id, op_map):

        for my_op, my_op_location, golden_op, golden_op_location in op_map:
            # 获取到所有需要比较的 tensor 的路径
            golden_tensor_paths = self.golden_data.getTensorPath(golden_token_id, golden_op, golden_op_location)
            my_tensor_paths = self.my_data.getTensorPath(my_token_id, my_op, my_op_location)
            
            # 交叉比对，记录结果
            for golden_tensor_path, my_tensor_path in itertools.product(golden_tensor_paths, my_tensor_paths):
                data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, data_id=0, token_id=golden_token_id)
                row_data = fill_row_data(data_info)
                self.compared_result.append(row_data)

    @classmethod
    def _flatten_and_enum_tuple(cls, *arr):
        for item in arr:
            if isinstance(item, (list, tuple)):
                yield from cls._flatten_enum_tuple(*item)
            else:
                yield item
