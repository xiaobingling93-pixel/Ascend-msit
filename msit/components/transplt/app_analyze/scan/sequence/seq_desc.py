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

import difflib
import multiprocessing as mp
from multiprocessing import Value
from app_analyze.utils.log_util import logger
from app_analyze.common.kit_config import KitConfig

_GLOBAl_FUNC_ID_DICT = dict()
_FUNC_ID_COUNTER_DICT = dict()

_GLOBAL_FILE_ID_DICT = dict()
_FILE_ID_COUNTER = Value('i', 0)


class FuncDesc:
    USR_DEFAULT_ID = -1

    def __init__(self):
        self.obj_info = None
        self.location = None
        self.acc_name = ''
        self.namespace = ''
        self.func_name = ''
        self.parm_num = 0
        self.parm_decl_names = list()
        self.return_type = ''
        self.is_usr_def = False
        self.root_file = None  # which file function define

        self.hash_code = 0
        self.is_cxx_method = False

        self._func_id = None

    @property
    def unique_name(self):
        full_str = self.full_name
        if self.acc_name:
            full_str = '[' + self.acc_name + ']' + full_str
        return full_str

    @property
    def full_name(self):
        api_str = self.api_name
        arg_str = self.arg_name
        api_str += '(' + arg_str + ')'
        return api_str

    @property
    def api_name(self):
        if self.obj_info:
            names = [self.obj_info.record_name, self.func_name]
        else:
            if self.namespace:
                names = [self.namespace, self.func_name]
            else:
                names = [self.func_name]

        return '::'.join(names)

    @property
    def arg_name(self):
        return ','.join(self.parm_decl_names)

    @property
    def func_id(self):
        return self._func_id

    @property
    def file_id(self):
        if not self.is_usr_def:
            return FuncDesc.USR_DEFAULT_ID

        fid = _GLOBAL_FILE_ID_DICT.get(self.root_file, None)
        if fid is None:
            fid = _FILE_ID_COUNTER.value
            _FILE_ID_COUNTER.value += 1
            _GLOBAL_FILE_ID_DICT[self.root_file] = fid
        return fid

    def set_func_id(self):
        if self.is_usr_def:
            self._func_id = FuncDesc.USR_DEFAULT_ID
            return

        name_id_tbl = _GLOBAl_FUNC_ID_DICT[self.acc_name]
        self._func_id = name_id_tbl.get(self.full_name, None)
        if self._func_id is not None:
            return

        rst = (None, None, 0)
        for fn_name, func_id in name_id_tbl.items():
            if not fn_name.startswith(self.api_name):
                continue

            new_params = fn_name.replace(self.api_name, '').strip('(').strip(')').split(',')
            if len(new_params) != self.parm_num:
                continue

            ratio = difflib.SequenceMatcher(None, fn_name, self.full_name).quick_ratio()
            if rst[2] < ratio:
                rst = (new_params, func_id, ratio)

        if rst[2] > 0:
            self.parm_decl_names = rst[0]
            self._func_id = rst[1]
            return

        base = KitConfig.ACC_LIB_ID_PREFIX[self.acc_name] * KitConfig.ACC_ID_BASE
        offset = _FUNC_ID_COUNTER_DICT[self.acc_name].value
        _FUNC_ID_COUNTER_DICT[self.acc_name].value = offset + 1
        self._func_id = base + offset
        _GLOBAl_FUNC_ID_DICT[self.acc_name][self.full_name] = self._func_id


class ObjDesc:
    def __init__(self):
        self.record_name = ''
        self.bases_num = 0
        self.is_polymorphic = False


class SeqDesc:
    def __init__(self):
        self.seq_id = 0
        self.seq = ''
        self.seq_count = 0
        self.entry_api = ''
        self.api_seq = list()
        self.has_usr_def = False
        self.has_called = False

    def trans_to(self):
        new_desc = SeqDesc()
        new_desc.entry_api = self.api_seq[0]
        new_desc.api_seq.extend(self.api_seq[1:])
        new_desc.has_usr_def = self.has_usr_def
        self.clear()
        return new_desc

    def clear(self):
        self.api_seq = list()
        self.has_usr_def = False

    def debug_string(self):
        rst = 'Entry Function is: ' + self.entry_api.api_name + '\n'
        apis = [_.full_name for _ in self.api_seq]
        rst += '-->'.join(apis)
        logger.debug(rst)


def get_idx_tbl():
    idx_dict = dict()
    for _, val in _GLOBAl_FUNC_ID_DICT.items():
        idx_dict.update(dict(zip(val.values(), val.keys())))

    return idx_dict


def get_api_lut():
    return _GLOBAl_FUNC_ID_DICT


def set_api_lut(idx_dict=None):
    for acc_lib, _ in KitConfig.ACC_LIB_ID_PREFIX.items():
        _GLOBAl_FUNC_ID_DICT[acc_lib] = mp.Manager().dict()
        _FUNC_ID_COUNTER_DICT[acc_lib] = Value('i', 0)

    if idx_dict is not None:
        acc_libs = set()
        base_id_dict = dict(zip(KitConfig.ACC_LIB_ID_PREFIX.values(), KitConfig.ACC_LIB_ID_PREFIX.keys()))
        for idx, name in idx_dict.items():
            res = int(idx) // KitConfig.ACC_ID_BASE
            acc_name = base_id_dict[res]
            acc_libs.add(acc_name)

            _GLOBAl_FUNC_ID_DICT[acc_name][name] = idx

        for acc_name in acc_libs:
            api_dict = _GLOBAl_FUNC_ID_DICT[acc_name]
            max_idx = max(api_dict.values())
            idx = int(max_idx) % KitConfig.ACC_ID_BASE
            _FUNC_ID_COUNTER_DICT[acc_name].value = idx + 1
