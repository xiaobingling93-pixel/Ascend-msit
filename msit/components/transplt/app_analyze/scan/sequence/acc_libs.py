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

from app_analyze.common.kit_config import KitConfig
from app_analyze.scan.sequence.seq_desc import get_idx_tbl

_GLOBAl_EXPERT_LIBS_DICT = dict()


def from_int(x):
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    raise Exception('Type error, x must be type int!')


def from_float(x):
    if isinstance(x, float) and not isinstance(x, bool):
        return x
    raise Exception('Type error, x must be type float!')


def from_str(x):
    if isinstance(x, str):
        return x
    raise Exception('Type error, x must be type str!')


def from_list(f, x):
    if isinstance(x, list):
        return [f(y) for y in x]
    raise Exception('Type error, x must be type list!')


def to_class(c, x):
    if isinstance(x, c):
        return x.to_dict()
    raise Exception('Type error, x must be type class!')


def to_enum(c, x):
    if isinstance(x, c):
        return x.value
    raise Exception('Type error, x must be type enum!')


def from_none(x):
    if x is None:
        return x
    raise Exception('Value error, x must be None!')


class Seq:
    def __init__(self, label, src_seq, seq_desc, dst_seqs):
        self.label = label
        self.src_seq = src_seq
        self.seq_desc = seq_desc
        self.dst_seqs = dst_seqs

    @staticmethod
    def from_dict(obj):
        if not isinstance(obj, dict):
            raise Exception('Type error, obj must be type dict!')

        label = from_str(obj.get('label'))
        src_seq = from_list(lambda x: from_int(x), obj.get('src_seq'))
        seq_desc = from_list(lambda x: from_str(x), obj.get('seq_desc'))
        dst_seqs = from_list(lambda x: from_list(from_int, x), obj.get('dst_seqs'))
        return Seq(label, src_seq, seq_desc, dst_seqs)

    def to_dict(self):
        result = dict()
        result['label'] = from_str(self.label)

        api_idx_dict = get_idx_tbl()
        src_seq = [api_idx_dict[_] for _ in self.src_seq]
        result['src_seq'] = from_list(lambda x: from_str(x), src_seq)

        result['seq_desc'] = from_list(lambda x: from_str(x), self.seq_desc)

        dst_seqs = list()
        for dst_seq in self.dst_seqs:
            dst_seqs.append([api_idx_dict[_] for _ in dst_seq])
        result['dst_seqs'] = from_list(lambda x: from_list(from_str, x), dst_seqs)

        return result


class SeqInfo:
    def __init__(self, seqs):
        self.seqs = seqs

    @staticmethod
    def from_dict(obj):
        if not isinstance(obj, dict):
            raise Exception('Type error, obj must be type dict!')

        seqs = from_list(Seq.from_dict, obj.get('seqs'))
        return SeqInfo(seqs)

    def to_dict(self):
        result = dict()
        result['seqs'] = from_list(lambda x: to_class(Seq, x), self.seqs)
        return result


class ExpertLibs:
    def __init__(self, acc_lib_dict):
        self.acc_lib_dict = acc_lib_dict

    @staticmethod
    def from_dict(obj):
        if not isinstance(obj, dict):
            raise Exception('Type error, obj must be type dict!')

        acc_lib_dict = dict()
        for lib, _ in KitConfig.ACC_LIB_ID_PREFIX.items():
            if obj.get(lib, None):
                lib_content = SeqInfo.from_dict(obj.get(lib))
                acc_lib_dict[lib] = lib_content

        return ExpertLibs(acc_lib_dict)

    def to_dict(self):
        result = dict()
        for lib, _ in KitConfig.ACC_LIB_ID_PREFIX.items():
            acc_lib = self.acc_lib_dict.get(lib, None)
            if acc_lib is not None:
                result[lib] = to_class(SeqInfo, acc_lib)
        return result


# load from json file
def expert_libs_from_dict(s):
    return ExpertLibs.from_dict(s)


# export json file, change api id to api name
def expert_libs_to_dict(x):
    return to_class(ExpertLibs, x)


def get_expert_libs():
    return expert_libs_from_dict(_GLOBAl_EXPERT_LIBS_DICT)


def set_expert_libs(expert_libs):
    _GLOBAl_EXPERT_LIBS_DICT.update(expert_libs)
