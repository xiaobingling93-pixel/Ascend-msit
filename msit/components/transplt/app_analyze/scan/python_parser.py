# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import time
from typing import Optional, Union

import jedi
import libcst
import libcst.matchers as m
import libcst.helpers as helper

from app_analyze.common.kit_config import KitConfig
from app_analyze.utils.log_util import logger
from app_analyze.utils.security import check_input_file_valid

RESULTS = list()


def get_file_content_bytes(file):
    check_input_file_valid(file)
    with open(file, 'rb') as file_handle:
        return file_handle.read()


def get_file_content(file):
    check_input_file_valid(file)
    with open(file, 'r', encoding='utf8') as file_handle:
        return file_handle.read()


class PythonAPIVisitor(libcst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        libcst.metadata.PositionProvider,
        libcst.metadata.QualifiedNameProvider,
        libcst.metadata.ParentNodeProvider,
        libcst.metadata.ScopeProvider,
    )

    def __init__(self, file, project_path, wrapper: libcst.metadata.MetadataWrapper):
        super(PythonAPIVisitor, self).__init__()
        self.file = file
        self.project_path = project_path
        self.wrapper = wrapper
        self.project = jedi.Project(path=self.project_path, added_sys_path=(KitConfig.ACC_PYTHON_LIB_FOLDER,))
        self.script = jedi.Script(code=get_file_content(self.file), path=self.file, project=self.project)

    @staticmethod
    def _is_acc_lib_api(api_name):
        for lib in KitConfig.API_MAP.keys():
            if api_name.startswith(lib):
                return True
        return False

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        api = self._get_full_name_for_node(node)
        if not self._is_acc_lib_api(api):
            return True
        cuda_en = True if "cuda" in api.lower() else False
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        loc = f"{self.file}, {position.start.line}:{position.start.column}"
        args = self._parse_args(node)
        item = {
            KitConfig.ACC_API: api,
            KitConfig.CUDA_EN: cuda_en,
            KitConfig.LOCATION: loc,
            KitConfig.CONTEXT: args,
            KitConfig.ACC_LIB: api.split(".")[0],
        }
        RESULTS.append(item)
        return True

    def _parse_args(self, node):
        parsed_args = []

        args = node.args
        func_definition_params = self._get_function_define_params_for_node(node)

        for (index, arg) in enumerate(args):
            assign_node = arg

            if m.matches(arg.value, m.SimpleString()):
                arg_real_name = arg.value.value
            elif not m.matches(arg.value, m.Name()):
                # current argument it's not a variable
                arg_real_name = self.wrapper.module.code_for_node(arg).rstrip(",")
            else:
                # get argument real name by finding assign node of current argument through ScopeProvider
                arg_real_name = arg.value.value
                scope = self.get_metadata(libcst.metadata.ScopeProvider, arg)
                for assignment in scope.assignments:
                    if assignment.name != arg_real_name:
                        continue

                    assign_node = self._get_parent_assign_node(assignment, arg_real_name)
                    break

            assign_code = self.wrapper.module.code_for_node(assign_node)
            position = self.get_metadata(libcst.metadata.PositionProvider, assign_node)
            assign_loc = f"{self.file}, {position.start.line}:{position.start.column}"
            arg_declaration_name = func_definition_params[index].name \
                if func_definition_params and len(func_definition_params) > index else "NaN"

            call_arg_str = f"{arg_declaration_name} | {arg_real_name} | {assign_code} | {assign_loc}"
            parsed_args.append(call_arg_str)

        return "\n".join(parsed_args)

    def _get_parent_assign_node(self, node, arg_name):
        if not isinstance(node, libcst.CSTNode) and hasattr(node, "node"):
            node = node.node

        # recursively finding parent node until meets a statement or module
        while not isinstance(node, libcst.SimpleStatementLine) and not isinstance(node, libcst.Module):
            if isinstance(node, libcst.metadata.scope_provider.ImportAssignment):
                # current argument it's assigned by an import statement
                return node.node
            elif isinstance(node, (libcst.Assign, libcst.Param)):
                # current argument it's assigned by an assigning a statement or an argument of parent function
                return node

            node = self.get_metadata(libcst.metadata.ParentNodeProvider, node)

        raise RuntimeError(f"Cannot find parent assign node for node {arg_name}")

    def _get_full_name_for_node(self, node: Union[str, libcst.CSTNode]) -> Optional[str]:
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        name = helper.get_full_name_for_node(node)
        pos = 0
        if "." in name:
            pos = name.rindex(".") + 1
        infer_result = self.script.infer(position.start.line, position.start.column + pos)
        if infer_result and len(infer_result) > 0 and not infer_result[0].full_name.startswith("builtins"):
            return infer_result[0].full_name
        name_list = list(self.get_metadata(libcst.metadata.QualifiedNameProvider, node))
        if name_list:
            return name_list[0].name
        return name

    def _get_function_define_params_for_node(self, node):
        code = self.wrapper.module.code_for_node(node)
        position = self.get_metadata(libcst.metadata.PositionProvider, node)
        bracket_pos = code.index("(")
        sig = self.script.get_signatures(position.start.line, position.start.column + bracket_pos + 1)
        if sig and len(sig) > 0:
            return sig[0].params
        return []


class Parser:
    # creates the object, does the initial parse
    def __init__(self, path, project_directory):
        logger.info(f'Scanning file: {path}')
        self.file = path
        self.project_directory = project_directory
        code = get_file_content_bytes(self.file)
        self.wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(code))
        self.api_visitor = PythonAPIVisitor(self.file, self.project_directory, self.wrapper)

    def parse(self):
        global RESULTS
        RESULTS.clear()

        start = time.time()
        self.wrapper.visit(self.api_visitor)

        logger.debug(f'Time elapsed： {time.time() - start:.3f}s')
        return RESULTS
