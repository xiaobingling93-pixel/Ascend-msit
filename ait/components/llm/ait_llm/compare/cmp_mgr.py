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
import torch

from ait_llm.common.log import logger
from ait_llm.compare.cmp_op_match import OpMatchMgr
from ait_llm.compare.cmp_data_parse import CompareDataParse, CompareDataTorch, CompareDataATB
from ait_llm.compare.cmp_utils import BasicDataInfo, fill_row_data, save_compare_reault_to_csv
from ait_llm.compare.multi_block import get_multi_tensor_paths, get_cat_dim


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

        op_map = list(self.op_match.match(self.golden_data, self.my_data))
        # 同步校验tokenid
        golden_tokens = self.golden_data.get_cmp_tokens()
        my_tokens = self.my_data.get_cmp_tokens()
        golden_token_set = set(self._flatten_and_enum_tuple(golden_tokens))
        my_token_set = set(self._flatten_and_enum_tuple(my_tokens))

        logger.debug(
            "compare tokens info ==> golden_tokens: %s, golden_token_set: %s \n my_tokens: %s, my_token_set: %s ",
            str(golden_tokens),
            str(golden_token_set),
            str(my_tokens),
            str(my_token_set),
        )
        if not golden_token_set.isdisjoint(my_token_set):
            cmp_tokens = golden_token_set.intersection(my_token_set)

            logger.debug("compare tokens is %s ", str(cmp_tokens))
            for token_id in cmp_tokens:
                logger.debug("comparing tokens is %s ", str(token_id))
                self.compare_token(token_id, token_id, op_map)
                logger.debug("compared tokens is %s ", str(token_id))
        else:
            if self.golden_data.get_token_id() is not None and self.my_data.get_token_id() is not None:
                # 指定token，强制比对
                for golden_token, my_token in itertools.product(golden_tokens, my_tokens):
                    self.compare_token(golden_token, my_token, op_map)
            else:
                logger.error(
                    f"my tokens is {my_tokens} and golden tokens is {golden_tokens}. The two cannot be matched."
                )
                return None

        return save_compare_reault_to_csv(self.compared_result, output_path)

    def compare_token(self, golden_token_id, my_token_id, op_map):
        for my_op, my_op_location, golden_op, golden_op_location in op_map:
            logger.debug("------ compare (%s %s)------", str(my_op.node_name), str(golden_op.node_name))
            # 获取到所有需要比较的 tensor 的路径
            golden_tensor_paths = list(self.golden_data.get_tensor_path(golden_token_id, golden_op, golden_op_location))
            my_tensor_paths = list(self.my_data.get_tensor_path(my_token_id, my_op, my_op_location))

            if len(golden_tensor_paths) == len(my_tensor_paths):
                golden_tensor_paths.sort()
                my_tensor_paths.sort()
                tensor_pairs = zip(golden_tensor_paths, my_tensor_paths)
            else:
                tensor_pairs = itertools.product(golden_tensor_paths, my_tensor_paths)

            # 交叉比对，记录结果
            for golden_tensor_path, my_tensor_path in tensor_pairs:
                logger.debug("golden_path: %s; my_path:%s", str(golden_tensor_path), str(my_tensor_path))

                # 1. get tensor_datas 多卡数据合并
                _, my_tensor_datas = get_multi_tensor_paths(
                    self.golden_data.get_token_path(golden_token_id), my_tensor_path, tensor_sub_dir=""
                )
                _, golden_tensor_datas = get_multi_tensor_paths(
                    self.golden_data.get_token_path(my_token_id), golden_tensor_path, tensor_sub_dir=""
                )
                # 2. concat tensor_datas
                dim_atb = get_cat_dim(my_tensor_datas, golden_tensor_datas)
                dim_torch = get_cat_dim(golden_tensor_datas, my_tensor_datas)
                atb_tensor_data = my_tensor_datas[0] if dim_atb == -1 else torch.cat(my_tensor_datas, dim_atb)
                torch_tensor_data = (
                    golden_tensor_datas[0] if dim_torch == -1 else torch.cat(golden_tensor_datas, dim_torch)
                )
                # 3. compare tensor_datas
                data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, token_id=my_token_id, data_id=0)
                row_data = fill_row_data(data_info, atb_tensor_data, torch_tensor_data)
                self.compared_result.append(row_data)

    @classmethod
    def _flatten_and_enum_tuple(cls, *arr):
        for item in arr:
            if isinstance(item, (list, tuple)):
                yield from cls._flatten_and_enum_tuple(*item)
            else:
                yield item
