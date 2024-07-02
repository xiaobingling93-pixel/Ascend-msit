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

from enum import Enum, unique
from app_analyze.common.kit_config import SeqArgs


@unique
class MatcherMode(Enum):
    NO_MATCHED = -1
    MATCHED = 0
    CONTAIN = 1
    BE_CONTAINED = 2
    CROSS = 3


# a:描出的序列; b:模板序列
def _calc_dist(a, b):
    seq = set(list(a.intersection(b)))
    num = len(seq)
    if len(b) == 0:
        raise Exception('Expert libs had no source sequence, please check expert_libs.json!')

    ratio = num * 1.0 / len(b)

    if len(a) == len(b) == num:
        mode = MatcherMode.MATCHED
    elif len(b) == num:
        # 表示模板序列包含在扫描出来的序列中
        mode = MatcherMode.CONTAIN
    elif len(a) == num:
        # 表示扫描出来的序列包含在模板序列中
        mode = MatcherMode.BE_CONTAINED
    elif num > 0:
        mode = MatcherMode.CROSS
    else:
        mode = MatcherMode.NO_MATCHED

    return ratio, mode, seq


def _filter(cur_acc_lib, cur_sim, cur_mt_seq, saved_seqs):
    def _compare(cur_mt_tuple, acc_mt_tuple):
        cur_mt_len, cur_val = cur_mt_tuple[0], cur_mt_tuple[1]
        acc_mt_len, acc_val = acc_mt_tuple[0], acc_mt_tuple[1]

        if cur_mt_len < acc_mt_len:
            flag = False
        elif cur_mt_len == acc_mt_len:
            flag = False if cur_val <= acc_val else True
        else:
            flag = True
        return flag

    valid_flag = True
    rm_keys = list()
    for acc_lib, acc_tuple in saved_seqs.items():
        acc_mt_seq = acc_tuple[1]
        _, mode, _ = _calc_dist(cur_mt_seq, acc_mt_seq)
        if mode in [MatcherMode.NO_MATCHED, MatcherMode.CROSS]:
            continue

        valid_flag = _compare((len(cur_mt_seq), cur_sim), (len(acc_mt_seq), acc_tuple[0]))
        if valid_flag:
            rm_keys.append(acc_lib)
        else:
            break

    [saved_seqs.pop(k) for k in rm_keys]
    if valid_flag:
        saved_seqs[cur_acc_lib] = (cur_sim, cur_mt_seq)


def _match(seq, acc_lib_seqs, rs_lib_seqs):
    # 遍历对应加速库的模板
    for lib_seq in acc_lib_seqs:
        # 获取加速库的模板序列
        src_seq = set(lib_seq.src_seq)
        # 计算扫描出的序列跟模板序列的匹配度
        val, mode, mt_seq = _calc_dist(seq, src_seq)

        # 小于阈值的，不进行推荐,SeqArgs.SIM_MIN_SUPPORT
        if val < SeqArgs.SIM_MIN_SUPPORT:
            continue

        if mode == MatcherMode.MATCHED:
            rs_lib_seqs.clear()
            rs_lib_seqs[lib_seq] = (val, mt_seq)
            return

        _filter(lib_seq, val, mt_seq, rs_lib_seqs)


def match_api_seqs(seqs, expert_libs):
    def _traverse_expert_libs(cur_seq_desc):
        mt_rst = dict()
        # 获取当前序列的id列表
        cur_idx_list = [_.func_id for _ in cur_seq_desc.api_seq if not _.is_usr_def]
        matched_ids = set(cur_idx_list)

        # 获取序列中所涉及的加速库
        acc_names = set([_.acc_name for _ in cur_seq_desc.api_seq if _.acc_name])
        for acc_name in acc_names:
            # 获取当前加速库对应的模板
            seq_info = expert_libs.acc_lib_dict.get(acc_name, None)
            if seq_info:
                # 进行模板匹配
                _match(matched_ids, seq_info.seqs, mt_rst)
        return mt_rst

    result = dict()

    for seq_desc in seqs:
        lib_seqs_rst = _traverse_expert_libs(seq_desc)
        # 对扫描出来的结果进行处理
        if lib_seqs_rst:
            lib_seqs_rst = dict(zip(lib_seqs_rst.keys(), [_[0] for _ in list(lib_seqs_rst.values())]))

        result[seq_desc] = lib_seqs_rst
    return result
