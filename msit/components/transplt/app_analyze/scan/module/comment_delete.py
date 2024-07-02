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


class CommentDelete:
    MULTI_COMMENT_C = ('/*', '*/')
    MULTI_COMMENT_CMAKE = ('#[[', ']]')
    SINGLE_COMMENT_f = ('C', 'c', '*')

    def __init__(self, contents, single_comment, multi_comment=None):
        """
        初始化
        :param contents: 检测的内容
        :param single_comment: 单行注释字符
        :param multi_comment: 多行注释字符，元组传入，例如 ('/*', '*/')
        """
        self.contents = contents
        self.single_comment = single_comment
        self.multi_comment = multi_comment
        self.solve_head = 0
        self.multi_comment_begin_index = -1
        self.exclude_comments = []
        self.multi_start_flag = False
        self.multi_end_flag = True

    def get_pattern(self):
        """
        获取匹配的字符串表达式
        :return:
        """

        if self.multi_comment:
            return self._multi_comment_pattern()
        else:
            return self._single_comment_pattern()

    def solve_single_comment(self, begin):
        """
        处理单行注释
        :param begin: 该注释所在的起始位置
        :return:
        """
        if self.multi_comment_begin_index != -1:
            # 在多行注释里面可能掺杂了单行注释
            # 例如：/*  abc  //abc  */
            return
        if begin > 0 and not self.contents[begin - 1].isspace() and self.single_comment != '!':
            # 判断注释符是不是代码的一部分
            # 例如：if test $$# -gt 0;
            return
        # 把注释符前面的先添加进来
        self.exclude_comments.append(self.contents[self.solve_head:begin])

        new_line = self.contents[begin:].find('\n')
        self.solve_head = begin + new_line
        if new_line == -1:
            # 当文件以注释符结尾的时候，则会找不到换行号，例如最后一行写的是:abc//
            self.solve_head = len(self.contents)

    def solve_multi_comment(self, kind, begin, end):
        """
        处理多行注释
        :param kind: 注释的种类
        :param begin: 注释的起始位置
        :param end: 注释的结束位置
        :return:
        """
        if not self.multi_comment:
            return

        if kind == self.multi_comment[0]:
            if self.multi_end_flag:
                self.exclude_comments.append(self.contents[self.solve_head:begin])
            else:
                if self.exclude_comments:
                    self.exclude_comments.pop()
                self.exclude_comments.append(self.contents[self.solve_head:begin])

            self.multi_comment_begin_index = begin
            self.multi_start_flag = True
            self.multi_end_flag = False

        elif kind == self.multi_comment[1]:
            if self.multi_start_flag:
                # 只保留换行符，其他的删除
                keep_newline = re.sub(r'[^\n]', '',
                                      self.contents
                                      [self.multi_comment_begin_index:end])
                self.solve_head = end
                self.exclude_comments.append(keep_newline)
                self.multi_comment_begin_index = -1
                self.multi_start_flag = False
                self.multi_end_flag = True

    def delete_comment(self):
        """
        删除注释的内容
        :return: 去掉注释后的内容
        """
        pattern = self.get_pattern()
        matches = re.finditer(pattern, self.contents, re.M)

        for match in matches:
            kind = match.group(0)
            begin = match.start()

            if begin < self.solve_head:
                continue

            if kind == self.single_comment:
                self.solve_single_comment(begin)
            elif isinstance(self.single_comment, tuple) and \
                    kind in self.single_comment:
                self.solve_single_comment(begin)
            else:
                self.solve_multi_comment(kind, begin, match.end())

        # 把最后一个注释后面的内容都加进来
        self.exclude_comments.append(self.contents[self.solve_head:])

        # 转换为字符串
        exclude_comments = ''.join(self.exclude_comments)
        return exclude_comments

    def _multi_comment_pattern(self):
        pattern = r''
        if "*" in self.multi_comment[0]:
            pattern = r'(%s)|(%s)|(%s)' % (
                self.multi_comment[0].replace('*', r'\*'),
                self.multi_comment[1].replace('*', r'\*'),
                self.single_comment)
        if "#" in self.multi_comment[0]:
            pattern = r'(%s)|(%s)|(%s)' % (
                self.multi_comment[0].replace('[', r'\['),
                self.multi_comment[1].replace(']', r'\]'),
                self.single_comment)
        return pattern

    def _single_comment_pattern(self):
        if self.single_comment == '!':
            # ' !DEC$ ' 和'!=' 后的内容不为注释，其余场景都是注释
            return r'(?=\s|'')(%s)(?!=|DEC\$ )' % self.single_comment
        elif self.single_comment == CommentDelete.SINGLE_COMMENT_f:
            return r'(?:^)(%s)(?=.*)' % 'C|c|\*'

        return r'(?=\s|'')(%s)' % self.single_comment
