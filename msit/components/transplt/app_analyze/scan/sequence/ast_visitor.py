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
from enum import Enum, unique
from clang.cindex import CursorKind
from app_analyze.common.kit_config import KitConfig
from app_analyze.scan.sequence.seq_desc import FuncDesc, ObjDesc
from app_analyze.scan.sequence.seq_utils import save_api_seq, reorder_args_apis, is_unused_api, rename_func_name
from app_analyze.scan.sequence.api_filter import GLOBAL_FILTER_PREFIX
from app_analyze.scan.clang_utils import call_expr, get_attr, get_children
from app_analyze.scan.clang_parser import cuda_enabled, usr_namespace, find_right_angle


@unique
class APIType(Enum):
    """
    api type
    """
    INVALID = 'invalid'
    USR_DEFINED = 'usr-defined'
    ACC_LIB = 'acc-lib'


# three kinds: 1.invalid, 2.usr_defined, 3.acc_lib
def _get_api_type(file, cursor):
    """判断该文件是否为加速库文件。"""
    arg_dict = {'cuda_en': False, 'usr_ns': '', 'api_type': APIType.INVALID}
    if not file:
        return arg_dict

    for lib, v in KitConfig.ACC_LIBS.items():
        if lib not in file:
            continue

        # 待ACC_LIBS的Pattern改为全路径后，可以使用file.startswith(lib)
        if v:
            # get relative path
            new_file = file if not file.startswith(lib) else file.replace(lib, '')
            arg_dict['cuda_en'] = cuda_enabled(new_file, v[1])
            arg_dict['usr_ns'] = usr_namespace(cursor, v[0])
            arg_dict['api_type'] = APIType.ACC_LIB
            cursor.lib = v[3]
        return arg_dict

    if file.startswith(KitConfig.SOURCE_DIRECTORY):
        arg_dict['api_type'] = APIType.USR_DEFINED
    return arg_dict


# input arguments
def _get_input_args(node):
    parameters = list()

    ref = node.referenced
    if not ref:
        return parameters

    parameters = [f'{x.type.spelling}' for x in ref.get_arguments()]
    if not parameters and node.displayname:
        idx = node.displayname.find(node.spelling)
        if idx != -1:
            res = node.displayname[idx + len(node.spelling):].strip('(').strip(')')
            parameters = res.split(',')
    return parameters


# class or struct info
def _get_obj_info(node, func_attr):
    def _get_namespace(c):
        namespace = ''
        prefix_name = list()
        if c.mangled_name:
            parent = c.semantic_parent
        else:
            fn_def = c.get_definition()
            parent = fn_def.semantic_parent if fn_def is not None else None

        while parent is not None and parent.kind != CursorKind.TRANSLATION_UNIT:
            prefix_name.append(parent.spelling)
            parent = parent.semantic_parent

        if prefix_name:
            prefix_name.reverse()
            namespace = '::'.join(prefix_name)
            return namespace

        info = call_expr(node)
        if info.api.endswith('.' + info.spelling):
            namespace = info.api.replace('.' + info.spelling, '')
        elif info.api.endswith('->' + info.spelling):
            namespace = info.api.replace('->' + info.spelling, '')
        elif info.api.endswith('::' + info.spelling):
            namespace = info.api.replace('::' + info.spelling, '')
        return namespace

    obj = ObjDesc()
    obj.record_name = _get_namespace(node)
    if obj.record_name == '':
        obj = None

    func_attr.obj_info = obj


def _visit_function_decl(node, api_type, arg_dict=None):
    def _format_api(ns, api):
        if ns == '' or api == '':
            return api

        # 拆分模板类型，保留第一个类型，cv::Ptr<cv::cudacodec::VideoReader>
        left_bracket, right_bracket = api.find('<'), find_right_angle('>')
        if left_bracket != -1 and right_bracket != -1:
            api = api[:left_bracket] + api[right_bracket + 1:]

        ns_end = api.rfind('::')
        if ns_end == -1:  # api无命名空间
            api = f'{ns}::{api}'
        else:
            api_ns = api[:ns_end]
            ns_idx = ns.find(api_ns)
            if api_ns.startswith(ns):
                pass
            elif ns_idx == -1:  # api_ns不在ns里，当然也not ns.startswith(api_ns)，例如cv和Scalar::all
                api = f'{ns}::{api}'
            elif not ns.startswith(api_ns):  # api.startswith('')为True，例如cv::dnn和dnn::Net
                api = f'{ns[:ns_idx]}{api}'
        return api

    func_attr = _create_func_desc(node)
    func_attr.is_usr_def = True if api_type == APIType.USR_DEFINED else False
    if api_type == APIType.ACC_LIB:
        func_attr.acc_name = get_attr(node, 'lib')
        func_attr.func_name = _format_api(arg_dict['usr_ns'], node.spelling)
        rename_func_name(func_attr)
    else:
        _get_obj_info(node, func_attr)

    func_attr.root_file = node.referenced.location.file.name
    if is_unused_api(func_attr):
        func_attr = None
    else:
        func_attr.set_func_id()

    return func_attr


def _visit_cxx_method(node, api_type=APIType.INVALID):
    func_attr = _create_func_desc(node)
    _get_obj_info(node, func_attr)
    func_attr.is_cxx_method = True
    func_attr.is_usr_def = True if api_type == APIType.USR_DEFINED else False
    if api_type == APIType.ACC_LIB:
        func_attr.acc_name = get_attr(node, 'lib')
        rename_func_name(func_attr)

    func_attr.root_file = node.referenced.location.file.name
    if is_unused_api(func_attr):
        func_attr = None
    else:
        func_attr.set_func_id()

    return func_attr


def _visit_call_expr(node, rst, pth):
    for c in get_children(node):
        cursor_kind = c.kind
        if cursor_kind != CursorKind.CALL_EXPR:
            cur_path = pth
            _visit_call_expr(c, rst, cur_path)
            continue

        if not c.referenced:
            return

        ref_kind = c.referenced.kind.name
        if ref_kind not in ['CXX_METHOD', 'FUNCTION_DECL']:
            cur_path = pth
            _visit_call_expr(c, rst, cur_path)
            continue

        arg_dict = _get_api_type(c.referenced.location.file.name, c)
        api_type = arg_dict.get('api_type')
        if api_type == APIType.INVALID:
            cur_path = pth
            _visit_call_expr(c, rst, cur_path)
            continue

        if api_type == APIType.ACC_LIB and c.spelling.startswith(GLOBAL_FILTER_PREFIX):
            cur_path = pth
            _visit_call_expr(c, rst, cur_path)
            continue

        if ref_kind == 'CXX_METHOD':
            func_attr = _visit_cxx_method(c, api_type)
        else:
            func_attr = _visit_function_decl(c, api_type, arg_dict)

        if func_attr:
            cur_path = list()
            cur_path.extend(pth),
            cur_path.append(c)
            rst.append((func_attr, cur_path))
        else:
            cur_path = pth
        _visit_call_expr(c, rst, cur_path)


def _usr_def_fn(node, seq_desc):
    func_attr = _visit_function_decl(node, APIType.USR_DEFINED)
    seq_desc.api_seq.append(func_attr)
    seq_desc.has_usr_def = True
    return False


def _usr_def_obj(node, seq_desc, cursor_kind):
    skip_flag = False
    if not node.referenced:
        return skip_flag

    ref_kind = node.referenced.kind.name
    if ref_kind != cursor_kind.name:
        return skip_flag

    arg_dict = _get_api_type(node.referenced.location.file.name, node)
    api_type = arg_dict.get('api_type')
    if api_type == APIType.USR_DEFINED:
        # 用户自定义对象和成员变量
        func_attr = _visit_cxx_method(node, api_type)
        seq_desc.api_seq.append(func_attr)
        seq_desc.has_usr_def = True
    return skip_flag


def _usr_and_lib_call_fn(node, seq_desc, ):
    skip_flag = False
    if not node.referenced:
        return skip_flag

    ref_kind = node.referenced.kind.name
    if ref_kind not in ['CXX_METHOD', 'FUNCTION_DECL', 'FUNCTION_TEMPLATE']:
        return skip_flag

    arg_dict = _get_api_type(node.referenced.location.file.name, node)
    api_type = arg_dict.get('api_type')
    if api_type == APIType.INVALID:
        return skip_flag

    if api_type == APIType.ACC_LIB and node.spelling.startswith(GLOBAL_FILTER_PREFIX):
        return skip_flag

    if ref_kind == 'CXX_METHOD':
        func_attr = _visit_cxx_method(node, api_type)
    else:
        func_attr = _visit_function_decl(node, api_type, arg_dict)

    if not func_attr:
        return skip_flag

    if api_type == APIType.USR_DEFINED:
        seq_desc.has_usr_def = True

    skip_flag = True
    rst = [(func_attr, [node])]
    pth = [node]
    _visit_call_expr(node, rst, pth)

    rst_size = len(rst)
    if rst_size == 1:
        seq_desc.api_seq.append(rst[0][0])
    else:
        reorder_args_apis(rst)
        seq_desc.api_seq.extend([val[0] for val in rst])
    return skip_flag


def visit(node, seq_desc, result):
    skip_flag = False
    cursor_kind = node.kind
    if cursor_kind in [CursorKind.FUNCTION_DECL, CursorKind.FUNCTION_TEMPLATE]:
        # 用户自定义函数
        save_api_seq(seq_desc, result)
        skip_flag = _usr_def_fn(node, seq_desc)
    elif cursor_kind in [CursorKind.CONSTRUCTOR, CursorKind.CXX_METHOD, CursorKind.DESTRUCTOR]:
        # 对象和成员变量
        save_api_seq(seq_desc, result)
        skip_flag = _usr_def_obj(node, seq_desc, cursor_kind)
    elif cursor_kind == CursorKind.CALL_EXPR:
        # 函数调用
        skip_flag = _usr_and_lib_call_fn(node, seq_desc)

    return skip_flag


def _create_func_desc(node):
    func_attr = FuncDesc()
    func_attr.func_name = node.spelling
    func_attr.return_type = node.result_type.spelling
    func_attr.location = _init_location(node)
    func_attr.hash_code = node.hash

    func_attr.parm_decl_names = _get_input_args(node)
    func_attr.parm_num = len(func_attr.parm_decl_names)
    return func_attr


def _init_location(node):
    location = dict()
    location['column'] = node.location.column
    location['file'] = os.path.abspath(node.location.file.name)
    location['line'] = node.location.line
    location['offset'] = node.location.offset
    return location
