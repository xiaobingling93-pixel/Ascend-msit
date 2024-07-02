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

import re
import os
import time
from collections import namedtuple, OrderedDict

import pandas as pd
import numpy as np

from app_analyze.common.kit_config import KitConfig
from app_analyze.scan.scanner import Scanner
from app_analyze.scan import scanner_utils
from app_analyze.scan.module.comment_delete import CommentDelete
from app_analyze.utils.log_util import logger


class CMakeScanner(Scanner):
    """
    cmake扫描器的具体子类
    """
    PATTERN = r'^(?:(?P<data>(?P<key_word>(\w*?))(?P<data_inner>((?:\s*\()([^\)]+)))\)))'
    SAVE_VAR_INFO_INPUT = namedtuple('save_var_info_input',
                                     ['func_name', 'body', 'start_line', 'match_flag', 'var_def_dict'])

    def __init__(self, files):
        super().__init__(files)
        self.name = 'CMakelists.txt'
        self.var_rel_commands = ['set', 'find_file', 'find_library', 'find_path', 'aux_source_directory',
                                 'pkg_check_modules']
        self.marco_pattern = r'\$\{[^{}]*?\}'
        self.pkg_pattern = r'PkgConfig::([0-9a-zA-Z]+)'

    @staticmethod
    def _check_var_info(val, start_line, var_def_dict):
        locs = var_def_dict[val]
        lines = list(locs.keys())
        idx = np.searchsorted(lines, start_line)
        flag = locs[lines[idx - 1]]
        return flag

    @staticmethod
    def read_cmake_file_content(filepath):
        """
        功能：读取CMakelists.txt文件内容，并删除注释
        :param filepath:文件路径
        :return:去掉注释后的文件内容
        """
        with open(filepath, errors='ignore') as file_desc:
            try:
                contents = file_desc.read()
            except UnicodeDecodeError as err:
                logger.error('%s decode error. Only the utf-8 format is '
                             'supported. Except:%s.', filepath, err)
                contents = ""
        contents = CommentDelete(contents, '#', CommentDelete.MULTI_COMMENT_CMAKE).delete_comment()
        return contents

    def do_scan(self):
        start_time = time.time()
        result = self._do_cmake_scan_with_file()
        self.porting_results['cmake'] = result
        self.update_include_dirs()
        eval_time = time.time() - start_time

        if result:
            logger.info(f'Total time for scanning cmake files is {eval_time}s')

    def update_include_dirs(self):
        """
        add_subdirectory (source_dir [binary_dir] [EXCLUDE_FROM_ALL])
        include_directories ([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])
        """
        result = []
        set_customization_dir = {}
        if self.files:
            # 获取工程根目录，后续作为路径前缀，默认CMakeLists.txt[0]在项目根路径下
            root_dir = os.path.dirname(self.files[0])
            for file in self.files:
                # 对外部传入路径进行校验
                if not os.path.isfile(file):
                    continue
                # 获取自定义的set路径
                set_customization_dir = self._match_set_dir(file, set_customization_dir, root_dir)
                result_match = self._match_include_directories_paths(file, root_dir, result, set_customization_dir)
                result.extend(result_match)
        # 修改KitConfig.INCLUDES并去重
        result = list(set(result))
        for i, item in enumerate(result):
            key = "CMakeLists_{i}".format(i=i)
            KitConfig.INCLUDES[key] = item

    def _match_include_directories_paths(self, file, root_dir, result, set_customization_dir):
        """
        获取自定义include_directories的路径
        return:返回自定义include_directories的路径列表
        """
        pattern = r'include_directories\(([^()]+)\)'
        with open(file) as f:
            cmake_content = f.read()
        include_directories_pattern = re.compile(pattern, re.DOTALL)
        matches = include_directories_pattern.findall(cmake_content)
        if matches:
            result_path = self._match_paths(matches, root_dir, set_customization_dir)
            result.extend(result_path)
        return result

    def _match_paths(self, matches, root_dir, set_customization_dir):
        """
        匹配扫描CMakeLists.txt下的include_directories路径，目前考虑${PROJECT_SOURCE_DIR}和[AFTER|BEFORE] [SYSTEM]
        return:返回路径列表
        """
        result = []
        for match in matches:
            # path[0]匹配双引号中的内容，path[1]匹配单引号中的内容，path[2]匹配空格分隔多个路径
            paths = re.findall(r'(?:"([^"]+)"|\'([^\']+)\')|([^"\';\s]+)', match)
            paths = [path[0] or path[1] or path[2] for path in paths]
            for path in paths:
                # 先判断本身路径和加根目录后是否为有效路径
                result = self._is_exists_path(path, result)
                processed_path = os.path.join(root_dir, path)
                result = self._is_exists_path(processed_path, result)
                # 判断是否为${PROJECT_SOURCE_DIR}或${CMAKE_CURRENT_SOURCE_DIR}
                path = self._replace_pre_variables(path, root_dir)
                result = self._is_exists_path(path, result)
                # 判断是否为类似${TEXT_DIR}/include自定义的其他路径
                path = self._extract_variable_path(path, set_customization_dir)
                result = self._is_exists_path(path, result)
        return result

    @staticmethod
    def _extract_variable_path(path, set_customization_dir):
        """判断是否为类似${TEXT_DIR}/include自定义的其他路径"""
        pattern = re.compile(r'\$\{([^{}]+_DIR)}/(.*)')
        match = pattern.match(path)
        if match:
            variable_name, path_after_variable = match.groups()
            if variable_name in set_customization_dir:
                path = os.path.join(set_customization_dir[variable_name], path_after_variable)
        return path

    @staticmethod
    def _is_exists_path(path, result):
        """判断是否为有效路径，如果是则加入result列表中"""
        if os.path.exists(path):
            result.append(path)
        return result

    def _match_set_dir(self, file, set_customization_dir, root_dir):
        """匹配set的库路径，匹配的格式为set(XXXX_DIR /xxx/xxxx/xxxx)"""
        set_pattern = re.compile(r'^set\((\w+_DIR)\s{1,20}([^)]+)\)')
        get_filename_pattern = re.compile(r'^get_filename_component\((\w+_DIR)\s+([^\s()]+)\)')
        with open(file) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                # 查找匹配的set语句
                match_set = set_pattern.match(line)
                # 查找匹配的get_filename_pattern语句
                match_getfilename = get_filename_pattern.match(line)
                if match_set:
                    variable_name, variable_value = match_set.groups()
                    variable_value = self._replace_pre_variables(variable_value, root_dir)
                    set_customization_dir[variable_name] = variable_value
                if match_getfilename:
                    variable_name, variable_value = match_getfilename.groups()
                    variable_value = self._replace_pre_variables(variable_value, root_dir)
                    set_customization_dir[variable_name] = variable_value
        return set_customization_dir

    @staticmethod
    def _replace_pre_variables(variable_value, root_dir):
        """替换其中的${CMAKE_CURRENT_SOURCE_DIR}和${PROJECT_SOURCE_DIR}为根目录路径"""
        if variable_value.startswith("${CMAKE_CURRENT_SOURCE_DIR}"):
            variable_value = os.path.join(root_dir, variable_value[len("${CMAKE_CURRENT_SOURCE_DIR}"):].lstrip("/"))
            return variable_value
        else:
            pattern = re.compile(r'\$\{([^{}]+)_SOURCE_DIR}/?(.*)')
            variable_value = pattern.sub(lambda match: os.path.join(root_dir, match.group(2)), variable_value)
        return variable_value

    def _do_cmake_scan_with_file(self):
        """
        功能：cmake全量扫描
        :return:
        """
        # 全量扫描对cmake添加编译选项只需对根路径的CMakelist.txt处理
        result = {}
        for file in self.files:
            # 对外部传入路径进行校验
            if not os.path.isfile(file):
                continue
            logger.info(f"Scanning file: {file}.")
            rst_vals = self._scan_cmake_function(file)
            result[file] = pd.DataFrame.from_dict(rst_vals)

        return result

    def _scan_cmake_function(self, filepath):
        rst_dict = {}

        contents = self.read_cmake_file_content(filepath)
        content_list = contents.split(")")
        for content in content_list:
            sentence = content + ")"
            rst_dict.update(self._match_cmake_function(contents, sentence))

        return list(rst_dict.values())

    def _match_cmake_function(self, contents, sentence):
        rst_dict = {}
        var_def_dict = OrderedDict()

        match = re.finditer(CMakeScanner.PATTERN, sentence, re.M)

        for item in match:
            content = item['data']
            func_name = item['key_word']
            body = item['data_inner']

            start_line = contents.count("\n", 0, contents.index(content))

            match_flag = False
            if KitConfig.MACRO_PATTERN.search(content) or KitConfig.LIBRARY_PATTERN.search(
                    content) or KitConfig.FILE_PATTERN.search(content):
                # exact match
                rst = {'lineno': start_line, 'content': content, 'command': func_name, 'suggestion': 'modifying'}
                rst_dict[start_line] = rst

                match_flag = True
            elif self._check_var_ref_info(body, start_line, var_def_dict):
                # reference match
                rst = {'lineno': start_line, 'content': content, 'command': func_name, 'suggestion': 'modifying'}
                rst_dict[start_line] = rst
            elif KitConfig.KEYWORD_PATTERN.search(content):
                # fuzzy match
                rst = {'lineno': start_line, 'content': content, 'command': func_name, 'suggestion': 'modifying'}
                rst_dict[start_line] = rst
            # save variable definition
            save_var_info_input = CMakeScanner.SAVE_VAR_INFO_INPUT(
                func_name, body, start_line, match_flag, var_def_dict)
            self._save_var_info(save_var_info_input)

        return rst_dict

    def _check_var_ref_info(self, body, start_line, var_def_dict):
        macros = []
        vals = re.findall(self.marco_pattern, body)
        macros.extend(vals)

        vals = re.findall(self.pkg_pattern, body)
        macros.extend(vals)

        for macro in macros:
            if var_def_dict.get(macro) is None:
                continue

            if self._check_var_info(macro, start_line, var_def_dict):
                return True

        return False

    def _save_var_info(self, save_var_info_input):
        func_name, body, start_line, match_flag, var_def_dict = save_var_info_input
        if func_name in self.var_rel_commands:
            # var define
            words = body.replace('(', '').strip().split(' ')
            if func_name == 'aux_source_directory':
                var_name = words[-1]
            else:
                var_name = words[0]

            if var_def_dict.get(var_name) is None:
                var_def_dict[var_name] = {start_line: match_flag}
            else:
                var_def_dict[var_name][start_line] = match_flag
