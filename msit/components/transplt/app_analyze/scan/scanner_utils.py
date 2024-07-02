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

def get_line_number(contents, elem):
    """
    获取匹配项所在的行号
    :param contents: 匹配全文
    :param elem: 匹配项
    :return: 行号起止值(从0开始)
    """
    start_lineno = contents[:elem.span()[0]].count('\n')
    end_lineno = contents[:elem.span()[1]].count('\n')

    return start_lineno, end_lineno


def get_content_dict(contents):
    """
    功能：获取源码文件中行号对应的内容的dict
    参数： contents 是删除注释后的内容
    返回:contents_dict 行号对应的内容
    """
    contents_dict = {}
    contents_list = contents.splitlines()
    num = 1
    for line in contents_list:
        if not line:
            line = "\n"
        contents_dict[num] = line
        num = num + 1
    return contents_dict
