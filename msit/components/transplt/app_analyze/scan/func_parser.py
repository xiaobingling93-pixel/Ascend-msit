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

import os
import time
from clang.cindex import CursorKind

from app_analyze.scan.clang_parser import Parser, macro_map, typedef_map, node_debug_string
from app_analyze.scan.clang_utils import get_attr, get_children, is_user_code
from app_analyze.scan.sequence.seq_desc import SeqDesc
from app_analyze.scan.sequence.seq_handler import SeqHandler
from app_analyze.scan.sequence.seq_utils import save_api_seq
from app_analyze.scan.sequence.ast_visitor import visit
from app_analyze.common.kit_config import KitConfig
from app_analyze.utils.log_util import logger


class FuncParser(Parser):
    def __init__(self, path):
        super().__init__(path)

    def _get_info(self, node, depth=0):
        children = [self._get_info(c, depth + 1) for c in node.get_children()]
        info = node_debug_string(node, children)
        return info

    def _parse_api(self, node, seq_desc, result):
        file = None
        if node.kind == CursorKind.TRANSLATION_UNIT:
            file = node.spelling
        else:
            if get_attr(node, 'location.file'):
                file = os.path.normpath(node.location.file.name)

        macro_map(node, file)
        typedef_map(node, file)

        skip_flag = False
        usr_code = is_user_code(file)
        if usr_code and not getattr(node, 'scanned', False):
            skip_flag = visit(node, seq_desc, result)

        children = list()
        if skip_flag:
            info = node_debug_string(node, children)
        else:
            for c in get_children(node):
                c_info = self._parse_api(c, seq_desc, result)
                if c_info:
                    children.append(c_info)

            info = None
            if usr_code:
                info = node_debug_string(node, children)

        return info

    @staticmethod
    def _handle_call_seqs(seqs):
        SeqHandler.union_api_seqs(seqs)

    def parse(self):
        for d in self.tu.diagnostics:
            if d.severity > KitConfig.TOLERANCE:
                logger.warning(f'Diagnostic severity {d.severity} > tolerance {KitConfig.TOLERANCE}, skip this file.')
                return dict()

        result = []

        seq_desc = SeqDesc()
        start = time.time()
        info = self._parse_api(self.tu.cursor, seq_desc, result)
        save_api_seq(seq_desc, result)
        logger.debug(f'Time elapsed： {time.time() - start:.3f}s')
        self._handle_call_seqs(result)

        return result
