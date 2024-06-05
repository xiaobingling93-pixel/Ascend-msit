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

    def getRootNode() -> list[TreeNode]:
        return None

    def getCmpToken(self) -> tuple:
        return tuple()

    def getTensorPath(self, token_id, node, location) -> tuple:
        return tuple()


class DataUtils:
    @staticmethod
    def get_token_ids(tokens_path, token_id):
        if token_id is not None:
            return (token_id,)
        if tokens_path is None:
            return tuple()
        return filter(lambda name: os.path.isdir(os.path.join(tokens_path, name)), os.listdir(tokens_path))


class CompareDataATB(CompareDataParse):
    def __init__(self, path, args) -> None:
        super().__init__(path, args)
        self.ait_dump_path = self.parse_ait_dump_path(path)
        self.token_id, self.pid, self.tokens_path = self.get_ids_by_path(self.ait_dump_path, path)
        self.token_ids = self._get_token_ids(self.tokens_path, self.token_id)
        self.topo_files = self.get_topo_file_path(self.ait_dump_path, self.pid)
        self.topo_trees = (self.parse(file) for file in self.topo_files)

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
        path_list = normpath.split(os.sep)
        MAX_PARSE_LEVEL = 3
        parse_level = 0
        for i in reversed(range(len(path_list))):
            if path_list[i].startsWith("ait_dump"):
                ait_dump_path = os.sep.join(path_list[0, i])
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
            tensor
                {device_id}_{pid} == tokens_path
                    {token_id}
        """
        rel_path = os.path.relpath(path, ait_dump_path)
        path_list = rel_path.split(os.sep)
        pid = None if len(path_list) < 2 else path_list[1].split("_")[-1]
        token_id = None if len(path_list) < 3 else path_list[2]
        tokens_path = (
            None if len(path_list) < 2 or path_list[0] != "tensor" else os.path.join(ait_dump_path, *path_list[0, 2])
        )
        if tokens_path is None:
            tensor_dir_path = os.path.join(ait_dump_path, "tensor")

            tokens_path_names = os.listdir(tensor_dir_path)
            tokens_path_names.sort()
            if len(tokens_path_names) == 0:
                return token_id, pid, tokens_path
            if pid is None:
                name = tokens_path_names[0]
                pid = name.split("_")[-1]
                tokens_path = os.path.join(tensor_dir_path, name)
            else:
                for name in tokens_path_names:
                    if pid == name.split("_")[-1]:
                        tokens_path = os.path.join(tensor_dir_path, name)
                        break

        return token_id, pid, tokens_path

    @staticmethod
    def get_model_path(ait_dump_path):
        return os.path.join(ait_dump_path, "model")

    @classmethod
    def get_topo_file_path(cls, ait_dump_path, pid=None):
        model_path = cls.get_model_path(ait_dump_path)
        if pid is None:
            pid_path_names = os.listdir(model_path)
            if len(pid_path_names) == 0:
                return None
            pid_path = os.path.join(model_path, pid_path_names[0])
        else:
            pid_path = os.path.join(model_path, pid)

        json_file_names = os.listdir()
        return (os.path.join(pid_path, name) for name in json_file_names)

    @staticmethod
    def load_topo_info(topo_file_path):
        with open(topo_file_path, "r") as file:
            topo_info = json.load(file)
        return topo_info

    def getRootNode(self) -> list[TreeNode]:
        return [x[0] for x in self.topo_trees]

    def getCmpToken(self) -> tuple:
        return self.token_ids

    def getTensorPath(self, token_id, node: TreeNode, location) -> tuple:
        token_path_id = token_id
        if len(self.topo_files) > 1:
            if isinstance(token_id, int):
                token_path_id = token_id - 1
            else:
                token_path_id = 0
        tensor_dir_path = node.tensor_path.replace("{token_id}", str(token_path_id))
        tensor_after_dir_path = os.path.join(tensor_dir_path, "after")
        tensor_before_dir_path = os.path.join(tensor_dir_path, "before")
        if location == MatchLocation.ALL_OUTPUT:
            if os.path.exists(tensor_after_dir_path):
                return (
                    os.path.join(tensor_after_dir_path, name)
                    for name in os.listdir(tensor_after_dir_path)
                    if name.startswith("outtensor")
                )
        elif location == MatchLocation.ALL_INPUT:
            if os.path.exists(tensor_after_dir_path):
                return (
                    os.path.join(tensor_after_dir_path, name)
                    for name in os.listdir(tensor_after_dir_path)
                    if name.startswith("intensor")
                )
            if os.path.exists(tensor_before_dir_path):
                return (
                    os.path.join(tensor_before_dir_path, name)
                    for name in os.listdir(tensor_before_dir_path)
                    if name.startswith("intensor")
                )
        else:
            return os.path.join(tensor_after_dir_path, location)

    def parse(self, topo_file: str) -> None:
        my_root_node = ModelTree.atb_json_to_tree(topo_file, os.path.join(self.tokens_path, "{token_id}"))
        my_layer_type = my_root_node.get_layer_node_type()
        logger.info("my_layer_type: %s", my_layer_type)
        my_layer_nodes = my_root_node.get_layer_node(my_layer_type)
        return my_root_node, my_layer_type, my_layer_nodes

    def _get_token_ids(self, tokens_path, token_id):
        if len(self.topo_files) > 1:
            ((0, 1) if t == 0 else t + 1 for t in DataUtils.get_token_ids(tokens_path, token_id))
        else:
            return DataUtils.get_token_ids(tokens_path, token_id)


class CompareDataTorch(CompareDataParse):
    def __init__(self, path, args) -> None:
        super().__init__(path, args)
        self.ait_dump_path = self.parse_ait_dump_path(path)
        self.token_id, self.pid, self.tokens_path = self.get_ids_by_path(self.ait_dump_path, path)
        self.token_ids = DataUtils.get_token_ids(self.tokens_path, self.token_id)
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

        return token_id, pid, tokens_path

    @staticmethod
    def get_topo_file_path(tokens_path: str):
        return os.path.join(tokens_path, "model_tree.json")

    def load_topo_info(self):
        topo_path = self.get_topo_file_path(self.path, self.args.cmp_level)
        with open(topo_path, "r") as file:
            topo_info = json.load(file)
        return topo_info

    def getRootNode(self) -> list[TreeNode]:
        return [self.golden_root_node]

    def getCmpToken(self) -> tuple:
        return self.token_ids

    def getTensorPath(self, token_id, node: TreeNode, location) -> tuple:
        tensor_dir_path = node.tensor_path.replace("{token_id}", str(token_id))
        if location == MatchLocation.ALL_OUTPUT:
            if os.path.exists(tensor_dir_path):
                return (
                    os.path.join(tensor_dir_path, name)
                    for name in os.listdir(tensor_dir_path)
                    if name.startswith("output")
                )
        elif location == MatchLocation.ALL_INPUT:
            if os.path.exists(tensor_dir_path):
                return (
                    os.path.join(tensor_dir_path, name)
                    for name in os.listdir(tensor_dir_path)
                    if name.startswith("input")
                )
        else:
            return os.path.join(tensor_dir_path, location)

    def parse(self) -> None:
        golden_root_node = ModelTree.json_to_tree(self.topo_file, os.path.join(self.tokens_path, "{token_id}"))
        golden_layer_type = golden_root_node.get_layer_node_type()
        logger.info("golden_layer_type: %s", golden_layer_type)
        golden_layer_nodes = golden_root_node.get_layer_node(golden_layer_type)
        return golden_root_node, golden_layer_type, golden_layer_nodes

    def get_tensor_path(
        self,
    ):
        pass
