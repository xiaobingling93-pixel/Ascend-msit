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

from prefixspan import PrefixSpan
from app_analyze.utils.log_util import logger
from app_analyze.utils.io_util import IOUtil
from app_analyze.common.kit_config import SeqArgs
from app_analyze.scan.sequence.seq_desc import get_idx_tbl


class SeqHandler:
    @staticmethod
    def union_api_seqs(seqs):
        def _get_union_api(seq_obj, api_seq):
            for api in seq_obj.api_seq:
                if not api.is_usr_def:
                    api_seq.append(api)
                else:
                    # if usr defined api in seq, check if api can union
                    usr_api = usr_def_dict.get(api.full_name, None)
                    if not usr_api:
                        api_seq.append(api)
                        continue

                    usr_api['key'].has_called = True
                    if seq_obj.entry_api.full_name == api.full_name:
                        # 递归函数
                        continue

                    _get_union_api(usr_api['key'], api_seq)

        if len(seqs) == 1:
            if all(not p.is_usr_def for p in seqs[0].api_seq):
                seqs[0].has_usr_def = False
            return

        usr_def_dict = dict()
        for seq_desc in seqs:
            usr_def_dict[seq_desc.entry_api.full_name] = {'key': seq_desc}

        for seq_desc in seqs:
            if not seq_desc.has_usr_def:
                continue

            new_api_seq = list()
            _get_union_api(seq_desc, new_api_seq)
            seq_desc.api_seq = new_api_seq
            if all(not p.is_usr_def for p in new_api_seq):
                seq_desc.has_usr_def = False

    @staticmethod
    def clean_api_seqs(seqs, infer_flag):
        def _compact_apis(api_seq):
            apis = list()
            pre_api_id = None
            for api in api_seq:
                if not pre_api_id:
                    pre_api_id = api.func_id
                    apis.append(api)
                elif pre_api_id == api.func_id:
                    continue
                else:
                    apis.append(api)
                    pre_api_id = api.func_id
            return apis

        # 序列去重
        def _dedup_apis(all_seqs):
            file_filter = dict()
            for seq_desc in all_seqs:
                entry_api = seq_desc.entry_api.full_name
                file = seq_desc.entry_api.location.get('file', None)
                entry_apis = file_filter.get(file, None)
                if entry_apis is None:
                    file_filter[file] = {entry_api: seq_desc}
                    continue

                item = entry_apis.get(entry_api, None)
                if item is None:
                    file_filter[file][entry_api] = seq_desc
                    continue

                if seq_desc.has_called:
                    file_filter[file][entry_api] = seq_desc

            res_seqs = list()
            for _, val_dict in file_filter.items():
                res_seqs += list(val_dict.values())
            return res_seqs

        # 同一个文件会被多个文件引用，所以序列有重复，需要根据文件过滤
        new_seqs = _dedup_apis(seqs)
        if infer_flag:
            rst = [seq_desc for seq_desc in new_seqs if not seq_desc.has_called]
            return rst

        rst = list()
        for seq_desc in new_seqs:
            if seq_desc.has_called:
                continue

            if not seq_desc.has_usr_def:
                # all acc lib apis in seq
                seq_desc.api_seq = _compact_apis(seq_desc.api_seq)
                rst.append(seq_desc)

                logger.debug(f'After clean seqs, api seqs length is {len(seq_desc.api_seq)}, the api seq is: ')
                seq_desc.debug_string()
                continue

            # delete use define api
            new_api_seq = [func_desc for func_desc in seq_desc.api_seq if not func_desc.is_usr_def]
            # deduplicate apis, eg: a b b b c --> a b c
            seq_desc.api_seq = _compact_apis(new_api_seq)
            seq_desc.has_usr_def = False
            rst.append(seq_desc)
            logger.debug(f'After clean seqs, api seqs length is {len(seq_desc.api_seq)}, the api seq is: ')
            seq_desc.debug_string()

        return rst

    @staticmethod
    def _store_api_seqs(seqs, id_dict=None, path='./'):
        seqs_file = path + 'seqs.tmp.json'
        IOUtil.json_safe_dump(seqs, seqs_file)

        seqs_idx_file = path + 'seqs_idx.tmp.json'
        if not id_dict:
            id_dict = get_idx_tbl()
        IOUtil.json_safe_dump(id_dict, seqs_idx_file)

    @staticmethod
    def debug_string(seqs, idx_dict=None):
        if not idx_dict:
            idx_dict = get_idx_tbl()

        rst_str = 'The sequences result are: \n'
        for i, seq in enumerate(seqs):
            d_str = [idx_dict[idx] for idx in seq]
            rst_str += str(i) + '. ' + '-->'.join(d_str) + '\n'

        logger.debug(f'{rst_str}')

    @staticmethod
    def mining_one_seq(seqs):
        def _len_two_lists(arr1, arr2):
            if not arr1 or not arr2:
                return 0
            idx = 0
            while idx < len(arr1) and idx < len(arr2) and arr1[idx] == arr2[idx]:
                idx += 1
            return idx

        def _longest_common_prefix(cur_seq_len, cur_array):
            for k in range(0, cur_seq_len - 1):
                for j in range(k + 1, cur_seq_len):
                    tmp = _len_two_lists(cur_array[k], cur_array[j])
                    if tmp <= 1:
                        continue

                    sub_rst = cur_array[k][0:tmp]
                    if sub_rst and len(sub_rst) >= SeqArgs.SEQ_MIN_LEN:
                        result.append(sub_rst)

        result = list()
        for seq in seqs:
            seq_len = len(seq)
            if seq_len <= 1:
                continue

            # 存放S的后缀字符串
            array = [seq[seq_len - 1 - i:] for i in range(0, seq_len)]

            # 两个相邻字符串的最长公共前缀
            _longest_common_prefix(seq_len, array)
        return result

    @staticmethod
    def mining_api_seqs(seqs):
        result = []
        ps = PrefixSpan(seqs)
        # second argument has 2 choice: closed and generator
        _ = ps.frequent(SeqArgs.PREFIX_SPAN_FREQ, closed=True)
        l1 = ps.topk(SeqArgs.PREFIX_SPAN_TOP_K, closed=True)

        for item in l1:
            seq = item[1]
            if len(seq) >= SeqArgs.SEQ_MIN_LEN:
                result.append(seq)
        return result

    def format_api_seqs(self, seqs):
        rst = []
        for seq_desc in seqs:
            cur_idx_list = [_.func_id for _ in seq_desc.api_seq]
            if cur_idx_list:
                rst.append(cur_idx_list)

        self._store_api_seqs(rst)
        return rst


def filter_api_seqs(seqs, idx_seq_dict=None):
    all_seqs = list()
    handler = SeqHandler()
    if not idx_seq_dict:
        seqs = handler.format_api_seqs(seqs)

    logger.debug('===============Sequences Before Filtering===============')
    handler.debug_string(seqs, idx_seq_dict)

    for seq in seqs:
        if len(seq) >= SeqArgs.SEQ_MIN_LEN:
            all_seqs.append(seq)

    logger.debug('===============Sequences After Filtering===============')
    handler.debug_string(all_seqs, idx_seq_dict)
    return all_seqs


def mining_api_seqs(seqs, idx_seq_dict=None):
    handler = SeqHandler()
    if not idx_seq_dict:
        seqs = handler.format_api_seqs(seqs)

    logger.debug('===============Sequences Before Mining===============')
    handler.debug_string(seqs, idx_seq_dict)

    all_seqs = set()
    dup_apis = handler.mining_one_seq(seqs)
    for apis in dup_apis:
        if len(set(apis)) == len(apis):
            all_seqs.add(tuple(apis))

    dig_apis = handler.mining_api_seqs(seqs)
    for apis in dig_apis:
        all_seqs.add(tuple(apis))

    logger.debug('===============Sequences After Mining===============')
    handler.debug_string(all_seqs, idx_seq_dict)
    return all_seqs
