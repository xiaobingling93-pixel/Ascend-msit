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

from abc import ABC, abstractmethod
import json
import os
from typing import List


from ait_llm.common.log import logger
from ait_llm.compare.atb_acc_cmp import is_model_topo_exist
from ait_llm.compare.cmp_op_match import MatchLocation
from ait_llm.dump.torch_dump.topo import ModelTree, TreeNode


class CompareDataParse(ABC):
    def __init__(self, path, args) -> None:
        super().__init__()
        self.args = args
        self.path = path

    @staticmethod
    @abstractmethod
    def accept(path: str) -> bool:
        return False

    def get_root_nodes(self) -> List[TreeNode]:
        return None

    def get_cmp_tokens(self) -> tuple:
        return tuple()

    def get_tensor_path(self, token_id, node, location) -> tuple:
        return tuple()


class DataUtils:
    @staticmethod
    def get_token_ids(tokens_path, token_id):
        if token_id is not None:
            return (token_id,)
        if tokens_path is None:
            return tuple()
        return filter(lambda name: os.path.isdir(os.path.join(tokens_path, name)), os.listdir(tokens_path))

    @staticmethod
    def get_dir_sub_files(dir_path, prefix):
        if os.path.exists(dir_path):
            return tuple((os.path.join(dir_path, name) for name in os.listdir(dir_path) if name.startswith(prefix)))
        else:
            return tuple()


class CompareDataATB(CompareDataParse):
    def __init__(self, path, args) -> None:
        super().__init__(path, args)
        self.ait_dump_path = self.parse_ait_dump_path(path)
        self.token_id, self.pid, self.tokens_path = self.get_ids_by_path(self.ait_dump_path, path)
        logger.debug(
            "atb input path is %s, \nait_dump_path is %s, \natb token_id is %s, \npid is %s, \ntokens_path is %s",
            str(path),
            str(self.ait_dump_path),
            str(self.token_id),
            str(self.pid),
            str(self.tokens_path),
        )
        self.topo_files = list(self.get_topo_file_path(self.ait_dump_path, self.pid))
        logger.debug("atb topo file is %s", str(self.topo_files))
        self.token_ids = list(self._get_token_ids(self.tokens_path, self.token_id))
        logger.debug("atb token ids is %s", str(self.token_ids))
        self.encode_root_node, self.decode_root_node = self.parse(self.topo_files)

    @classmethod
    def accept(cls, path: str, args) -> bool:
        ait_dump_path = cls.parse_ait_dump_path(path)

        if ait_dump_path and os.path.exists(path) and cls.get_topo_file_path(ait_dump_path) is not None:
            return True
        else:
            return False

    @staticmethod
    def parse_ait_dump_path(path):
        ait_dump_path = None

        normpath = os.path.normpath(path)
        path_list = list(normpath.split(os.sep))
        MAX_PARSE_LEVEL = 3
        parse_level = 0
        for index in reversed(range(len(path_list))):
            if path_list[index].startswith("ait_dump"):
                ait_dump_path = os.sep.join(path_list[0 : index + 1])
                break
            parse_level += 1
            if parse_level > MAX_PARSE_LEVEL:
                break

        return ait_dump_path

    @staticmethod
    def get_ids_by_path(ait_dump_path, path):
        """
        ait_dump_path
            model
                {pid}
                    {topo}.json
            tensors
                {device_id}_{pid} == tokens_path
                    {token_id}
        """
        rel_path = os.path.relpath(path, ait_dump_path)
        path_list = rel_path.split(os.sep)
        pid = None if len(path_list) < 2 else path_list[1].split("_")[-1]
        token_id = None if len(path_list) < 3 else path_list[2]
        tokens_path = (
            None if len(path_list) < 2 or path_list[0] != "tensors" else os.path.join(ait_dump_path, *path_list[0:2])
        )
        if tokens_path is None:
            tensor_dir_path = os.path.join(ait_dump_path, "tensors")

            tokens_path_names = os.listdir(tensor_dir_path)
            tokens_path_names.sort()
            if len(tokens_path_names) == 0:
                return token_id, pid, os.path.abspath(tokens_path)
            if pid is None:
                name = tokens_path_names[0]
                pid = name.split("_")[-1]
                tokens_path = os.path.join(tensor_dir_path, name)
            else:
                for name in tokens_path_names:
                    if pid == name.split("_")[-1]:
                        tokens_path = os.path.join(tensor_dir_path, name)
                        break

        return token_id, pid, os.path.abspath(tokens_path)

    @staticmethod
    def get_model_path(ait_dump_path):
        return os.path.join(ait_dump_path, "model")

    @classmethod
    def get_topo_file_path(cls, ait_dump_path, pid=None):
        model_path = cls.get_model_path(ait_dump_path)
        if pid is None:
            if os.path.exists(model_path):
                return None
            pid_path_names = os.listdir(model_path)
            if len(pid_path_names) == 0:
                return None
            pid_path = os.path.join(model_path, pid_path_names[0])
        else:
            pid_path = os.path.join(model_path, pid)

        json_file_names = os.listdir(pid_path)
        return (os.path.join(pid_path, name) for name in json_file_names if name.endswith(".json"))

    @staticmethod
    def load_topo_info(topo_file_path):
        with open(topo_file_path, "r") as file:
            topo_info = json.load(file)
        return topo_info

    def get_root_nodes(self) -> List[TreeNode]:
        if self.encode_root_node is None:
            return [self.decode_root_node]
        else:
            return [self.encode_root_node, self.decode_root_node]

    def get_cmp_tokens(self) -> tuple:
        return self.token_ids

    def get_tensor_path(self, token_id, node: TreeNode, location) -> tuple:
        token_path_id = token_id
        if len(self.topo_files) > 1:
            if isinstance(token_id, int):
                token_path_id = token_id - 1
            else:
                token_path_id = 0
        if isinstance(token_id, int) and token_path_id < 0:
            token_path_id = 0

        # 控制特定比较token 0 或 1 的时候，只比较 encode 或 decode 一个
        if token_path_id == 0 and self.encode_root_node is not None and len(self.decode_root_node.children) > 0:
            logger.debug(
                "token_path_id: %s, token_id: %s, node.order: %s, self.decode_root_node.order: %s",
                str(token_path_id),
                str(token_id),
                str(node.order),
                str(self.decode_root_node.children[0].order),
            )
            if token_id == 0 and node.order >= self.decode_root_node.children[0].order:
                return ()
            elif token_id == 1 and node.order < self.decode_root_node.children[0].order:
                return ()
            else:
                pass

        logger.debug(
            "atb get_tensor_path: token_id: %s, token_path_id： %s, node: %s, localtion: %s",
            str(token_id),
            str(token_path_id),
            str(node),
            str(location),
        )
        tensor_dir_path = node.tensor_path.replace("{token_id}", str(token_path_id))
        tensor_after_dir_path = os.path.join(tensor_dir_path, "after")
        tensor_before_dir_path = os.path.join(tensor_dir_path, "before")
        logger.debug(
            "atb get_tensor_path: tensor_dir_path: %s, tensor_after_dir_path %s, tensor_before_dir_path: %s",
            str(tensor_dir_path),
            str(tensor_after_dir_path),
            str(tensor_before_dir_path),
        )
        if location == MatchLocation.ALL_OUTPUT:
            return DataUtils.get_dir_sub_files(tensor_after_dir_path, "outtensor")
        elif location == MatchLocation.ALL_INPUT:
            return DataUtils.get_dir_sub_files(tensor_after_dir_path, "intensor") + DataUtils.get_dir_sub_files(
                tensor_before_dir_path, "intensor"
            )
        else:
            if os.path.exists(os.path.join(tensor_after_dir_path, location)):
                return (os.path.join(tensor_after_dir_path, location),)
            else:
                return tuple()

    def parse(self, topo_files) -> None:
        topo_infos = []
        for topo_file in topo_files:
            json_start_order = 0
            with open(topo_file, "r") as file:
                node_dict = json.loads(file.read(), parse_constant=lambda x: None)
                nodes = node_dict.get("nodes", [])
                if len(nodes) != 0:
                    start_order_str: str = nodes[0].get("opName", "0").split("_")[-1]
                    if start_order_str.isdigit():
                        json_start_order = int(start_order_str)
            topo_infos.append(dict(path=topo_file, json_start_order=json_start_order, node_cnt=len(nodes)))

        topo_infos.sort(key=lambda x: x.get("json_start_order"))

        # 特殊处理，存在transdata，编号往前一步
        transdata_operation_flag = os.path.exists(
            os.path.join(self.ait_dump_path, "layer", self.pid, "TransdataOperation_0.json")
        )
        transdata_order = 1 if transdata_operation_flag else 0
        my_root_nodes = []
        for topo_info in topo_infos:
            topo_file = topo_info.get("path")
            start_order = topo_info.get("json_start_order") + transdata_order
            logger.debug("parseing topo file: %s, and start order is: %d", topo_file, start_order)
            my_root_node = ModelTree.atb_json_to_tree(
                topo_file, os.path.join(self.tokens_path, "{token_id}"), start_order
            )
            my_root_nodes.append(my_root_node)

        if len(my_root_nodes) == 1:
            return [None, my_root_nodes[0]]
        return my_root_nodes

    def _get_token_ids(self, tokens_path, token_id):
        token_ids = DataUtils.get_token_ids(tokens_path, token_id)
        logger.debug("%s", str(token_ids))
        if len(self.topo_files) > 1:
            return ((0, 1) if t == '0' else int(t) + 1 if t.isdigit() else t for t in token_ids)
        else:
            return (int(t) if t.isdigit() else t for t in token_ids)


class CompareDataTorch(CompareDataParse):
    def __init__(self, path, args) -> None:
        super().__init__(path, args)
        self.ait_dump_path = self.parse_ait_dump_path(path)
        self.token_id, self.pid, self.tokens_path = self.get_ids_by_path(self.ait_dump_path, path)
        self.token_ids = [
            int(t) if t.isdigit() else t for t in DataUtils.get_token_ids(self.tokens_path, self.token_id)
        ]
        self.topo_file = self.get_topo_file_path(self.tokens_path)
        self.golden_root_node, self.golden_layer_type, self.golden_layer_nodes = self.parse()

    @classmethod
    def accept(cls, path: str, args) -> bool:
        ait_dump_path = cls.parse_ait_dump_path(path)
        return ait_dump_path is not None

    @staticmethod
    def parse_ait_dump_path(path):
        normpath = os.path.normpath(path)
        MAX_PARSE_LEVEL = 3
        for i in range(MAX_PARSE_LEVEL):
            up_level = [".."] * i
            ait_dump_path = os.path.join(normpath, *up_level)
            if os.path.exists(os.path.join(ait_dump_path, "model_tree.json")):
                return os.path.join(ait_dump_path, "..")
        return None

    @staticmethod
    def get_ids_by_path(ait_dump_path, path):
        """
        ait_dump_path
            {device_id}_{pid} == tokens_path
                model_tree.json
                {token_id}
        """
        rel_path = os.path.relpath(path, ait_dump_path)
        path_list = rel_path.split(os.sep)
        pid = None if len(path_list) < 1 else path_list[0].split("_")[-1]
        token_id = None if len(path_list) < 2 else path_list[1]
        tokens_path = None if len(path_list) < 1 else os.path.join(ait_dump_path, path_list[0])
        if tokens_path is None:
            tokens_path_names = os.listdir(ait_dump_path)
            tokens_path_names.sort()
            if len(tokens_path_names) == 0:
                return token_id, pid, tokens_path
            # 找第一个
            name = tokens_path_names[0]
            pid = name.split("_")[-1]
            tokens_path = os.path.join(ait_dump_path, name)

        return token_id, pid, os.path.abspath(tokens_path)

    @staticmethod
    def get_topo_file_path(tokens_path: str):
        return os.path.join(tokens_path, "model_tree.json")

    def load_topo_info(self):
        topo_path = self.get_topo_file_path(self.path, self.args.cmp_level)
        with open(topo_path, "r") as file:
            topo_info = json.load(file)
        return topo_info

    def get_root_nodes(self) -> List[TreeNode]:
        return [self.golden_root_node]

    def get_cmp_tokens(self) -> tuple:
        return self.token_ids

    def get_tensor_path(self, token_id, node: TreeNode, location) -> tuple:
        tensor_dir_path = node.tensor_path.replace("{token_id}", str(token_id))
        logger.debug(
            "get_tensor_path: token_id: %s, node: %s, localtion: %s, tensor_dir_path: %s",
            str(token_id),
            str(node),
            str(location),
            str(tensor_dir_path),
        )
        if location == MatchLocation.ALL_OUTPUT:
            return DataUtils.get_dir_sub_files(tensor_dir_path, "output")
        elif location == MatchLocation.ALL_INPUT:
            return DataUtils.get_dir_sub_files(tensor_dir_path, "input")
        else:
            if os.path.exists(os.path.join(tensor_dir_path, location)):
                return (os.path.join(tensor_dir_path, location),)
            else:
                return tuple()

    def parse(self) -> None:
        golden_root_node = ModelTree.json_to_tree(self.topo_file, os.path.join(self.tokens_path, "{token_id}"))
        golden_layer_type = golden_root_node.get_layer_node_type()
        logger.info("golden_layer_type: %s", golden_layer_type)
        golden_layer_nodes = golden_root_node.get_layer_node(golden_layer_type)
        return golden_root_node, golden_layer_type, golden_layer_nodes
