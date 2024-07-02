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

from app_analyze.scan.sequence.api_filter import ACC_FILTER


def is_unused_api(func_desc):
    val = ACC_FILTER.get(func_desc.acc_name, None)
    if val is None:
        return False

    acc_file = func_desc.root_file
    file_filter = val.get('file_filter', [])
    if any(acc_file.endswith(p) for p in file_filter):
        return True

    api_name = func_desc.api_name
    api_filter = val.get('api_filter', [])
    if api_name in api_filter:
        return True

    return False


def rename_func_name(func_desc):
    def _match_namespace_prefix(prefix):
        matched_flag = True
        idx = -1

        if func_name.startswith(prefix):
            idx = 0
        elif record_name.startswith(prefix):
            idx = 1
        else:
            matched_flag = False
        return matched_flag, idx

    def _set_name(idx, prefix):
        res = name_tuple[idx].replace(prefix, '')
        if res.startswith('::'):
            if not idx:
                func_desc.func_name = ns + res
            else:
                func_desc.obj_info.record_name = ns + res
        else:
            pos = res.find('::')
            if not idx:
                func_desc.func_name = ns + res[pos:]
            else:
                func_desc.obj_info.record_name = ns + res[pos:]

    val = ACC_FILTER.get(func_desc.acc_name, None)
    if val is None:
        return

    ns_filter = val.get('namespace_filter', {})
    func_name = func_desc.func_name
    record_name = func_desc.obj_info.record_name if func_desc.obj_info is not None else ''
    name_tuple = (func_name, record_name)
    for ns_prefix, ns in ns_filter.items():
        mt_flag, nm_id = _match_namespace_prefix(ns_prefix)
        if not mt_flag:
            continue

        _set_name(nm_id, ns_prefix)
        break


def save_api_seq(seq_desc, result):
    api_cnt = len(seq_desc.api_seq)
    if api_cnt == 1:
        seq_desc.clear()
    elif api_cnt > 1:
        new_seq_desc = seq_desc.trans_to()
        result.append(new_seq_desc)


def _reorder(idx_tuple, degree_tuple, flag_tuple, nodes):
    cur_idx, nxt_idx = idx_tuple[0], idx_tuple[1]
    cur_node_degree, nxt_node_degree = degree_tuple[0], degree_tuple[1]
    idx_flag, chk_flag = flag_tuple[0], flag_tuple[1]

    node_cnt = len(nodes)
    cur_node = nodes[cur_idx]
    stop_flag = False

    if cur_node_degree < nxt_node_degree:
        # 当前节点的深度比后续遍历节点的深度小
        if nxt_idx + 1 == node_cnt:
            # 比所有节点都小，该节点是根节点，要排在最后面
            nodes.pop(cur_idx)
            nodes.insert(nxt_idx, cur_node)
            # 当前节点向后移动，下一个节点的索引跟当前节点索引一样，不需要更改
            idx_flag = False
        else:
            # 继续向后移动，进行位置移动判断
            chk_flag = True
    else:
        if chk_flag:
            # 父节点需要向后移动
            nodes.pop(cur_idx)
            nodes.insert(nxt_idx - 1, cur_node)
            # 当前节点向后移动，下一个节点的索引跟当前节点索引一样，不需要更改
            idx_flag = False
        elif cur_node_degree == nxt_node_degree:
            # 深度一样，是兄弟节点，不需要进行位置移动
            stop_flag = True

    return idx_flag, chk_flag, stop_flag


# 对函数的参数的api进行逆排序
# eg: origin order is:
# [('s',['s']),('a',['s','a']),('c',['s','a','c']),('d',['s','a','d']),('b',['s','b']),('e',['s','b','e'])]
# after reorder, order is:
# [('c',['s','a','c']),('d',['s','a','d']),('a',['s','a']),('e',['s','b','e']),('b',['s','b']),('s',['s'])]
def reorder_args_apis(seq):
    idx = 0  # 当前需要遍历的节点的索引
    visited_apis = list()  # 已经遍历过的节点
    i = 0  # 遍历次数
    cnt = len(seq)  # 节点的个数

    while i < cnt:
        idx_moved_flag = True  # 遍历节点时，指针的位置是否需要移动
        check_nxt_flag = False  # 是否需要向后遍历

        # (func_desc, [node])
        api = seq[idx]
        if api in visited_apis:
            idx += 1
            continue

        # 将没有访问过的节点加入列表中
        visited_apis.append(api)
        # 当前节点的深度
        cur_node_degree: int = len(api[1])
        for j in range(idx + 1, cnt):
            # 对比节点的深度
            nxt_node_degree: int = len(seq[j][1])
            idx_moved_flag, check_nxt_flag, stop_flag = _reorder((idx, j), (cur_node_degree, nxt_node_degree),
                                                                 (idx_moved_flag, check_nxt_flag), seq)
            if not idx_moved_flag or stop_flag:
                break

        if idx_moved_flag:
            # 移动当前指针的位置
            idx += 1
        i += 1
