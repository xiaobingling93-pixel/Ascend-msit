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
import pandas as pd


class SeqAdvisor:
    def __init__(self, result, idx_dict):
        self.result = result
        self.api_idx_dict = idx_dict

    def recommend(self):
        def _format_matched_seq(mt_idx, src_tuple, dst_tuple, rst):
            usr_entry_api, usr_call_seq = src_tuple[0], src_tuple[1]
            dst_mt_lib, mt_rate = dst_tuple[0], dst_tuple[1]

            match_seq = '-->'.join([self.api_idx_dict[_] for _ in dst_mt_lib.src_seq])
            for j, dst_seq in enumerate(dst_mt_lib.dst_seqs):
                rec_seq = '-->'.join([self.api_idx_dict[_] for _ in dst_seq])
                if j == 0:
                    label = str(mt_idx + 1) + '.' + dst_mt_lib.label
                    if mt_idx == 0:
                        item = [usr_entry_api, usr_call_seq, label, rec_seq, dst_mt_lib.seq_desc[j], str(mt_rate),
                                match_seq]
                    else:
                        item = ['', usr_call_seq, label, rec_seq, dst_mt_lib.seq_desc[j], str(mt_rate), match_seq]
                else:
                    item = ['', '', '', rec_seq, dst_mt_lib.seq_desc[j], str(mt_rate), '']
                rst.append(item)

        data_dict = dict()
        for seq_desc, lib_seqs in self.result.items():
            content = []
            entry_api = seq_desc.entry_api.full_name
            src_seq = '-->'.join([_.full_name for _ in seq_desc.api_seq])

            i = 0
            for lib_seq, rate in lib_seqs.items():
                _format_matched_seq(i, (entry_api, src_seq), (lib_seq, rate), content)
                i += 1

            if not lib_seqs:
                content = [[entry_api, src_seq, '', '', '', '', '']]

            loc = seq_desc.entry_api.location['file']
            if data_dict.get(loc, None):
                data_dict[loc] += content
            else:
                data_dict[loc] = content

        df_dict = dict()
        for f, data in data_dict.items():
            df = pd.DataFrame(data,
                              columns=['Entry API', 'Usr Call Seqs', 'Seq Labels', 'Recommended Sequences',
                                       'Functional Description', 'Recommendation Index', 'Tmp Match Sequences'
                                       ],
                              dtype=str)
            df_dict[f] = df
        return df_dict

    @property
    def cxx_format_fn(self):
        return self._postprocess

    @property
    def common_format_fn(self):
        return self._dataframe_to_worksheet

    @staticmethod
    def _dataframe_to_worksheet(file_name, data, workbook):
        worksheet = workbook.add_worksheet(file_name)
        header = data.columns.values
        worksheet.write_row('A1', header[0:len(header)])
        for i in range(data.shape[0]):
            columns = [data.loc[i, _] for _ in header]
            worksheet.write_row('A' + str(i + 2), columns)

    @staticmethod
    def _postprocess(file_name, data, workbook):
        # 给特性字符添加转义字符，为了正则匹配做预处理
        def _escape_string(word):
            escape_seq = ['<', '>', '.', '*', '(', ')', '[', ']']
            result = word
            for s in escape_seq:
                result = result.replace(s, '\\' + s)
            return result

        # 根据匹配的api，对序列进行切分
        def _split_call_seq(mt_str, call_seq):
            keyword_sep = mt_str.strip('|')
            keyword_split = keyword_sep.split('|')  # 这样就按文本中出现关键词的顺序列出了

            # 用所有关键词将整段话分割，再插入富字符串，然后捆绑颜色、关键词和后面的文本，需注意一一对应
            temp_list = re.split(_escape_string(keyword_sep), call_seq)
            sep_str = list()
            for i, element in enumerate(temp_list):
                if i != 0:
                    if element != '':
                        sep_str.extend((fmt, keyword_split[i - 1], element))
                    else:
                        sep_str.extend((fmt, keyword_split[i - 1]))
                else:
                    if element != '':
                        sep_str.append(element)

            rich_string_flag = True
            if len(keyword_split) == 1 and keyword_sep == call_seq:
                # 扫描出来的api只有一个，并且有推荐的api
                rich_string_flag = False
            return rich_string_flag, sep_str

        worksheet = workbook.add_worksheet(file_name)
        # 设置excel格式
        fmt = workbook.add_format({'color': 'red'})
        # 获取表头字段
        header = data.columns.values
        # 写表头
        worksheet.write_row('A1', header[0:len(header) - 1])
        # 获取usr_call_seq所在列号
        idx = data.columns.get_loc('Usr Call Seqs')
        for num in range(data.shape[0]):
            # 获取表内容
            columns = [data.loc[num, _] for _ in header]
            mt_seq = columns.pop(-1)
            if not mt_seq:
                worksheet.write_row('A' + str(num + 2), columns)
                continue

            keywords = mt_seq.split('-->')
            src_call_seq = columns[idx]
            words = src_call_seq.split('-->')
            # 保存匹配到的关键词
            match_str = '|'.join(list(set(keywords).intersection(set(words))))

            # 虽然关键词已保存，但并未按其在文本中出现的位置顺序,通过re.finditer按文本中出现的顺序匹配，所以进行二次匹配
            re_match = re.finditer(_escape_string(match_str.strip('|')), src_call_seq)

            re_str = ''
            for m in re_match:
                re_str += str(m.group()) + '|'

            rich_flag, params = _split_call_seq(re_str, src_call_seq)
            for j, val in enumerate(columns):
                if j == idx:
                    if rich_flag:
                        worksheet.write_rich_string(num + 1, j, *params)
                    else:
                        worksheet.write(num + 1, j, val, fmt)
                else:
                    worksheet.write(num + 1, j, val)
