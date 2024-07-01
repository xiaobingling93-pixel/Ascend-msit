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

import logging
import re
import os
import platform
import time
from copy import deepcopy

from clang.cindex import Index, CursorKind, TranslationUnit, Config

from app_analyze.common.kit_config import KitConfig
from app_analyze.utils.io_util import IOUtil
from app_analyze.utils.log_util import logger
from app_analyze.utils.lib_util import is_acc_path
from app_analyze.scan.clang_utils import helper_dict, filter_dict, Info, get_attr, get_children, skip_implicit
from app_analyze.scan.clang_utils import auto_match, read_cursor, TYPEDEF_MAP, is_user_code

SCANNED_FILES = list()
RESULTS = list()
MACRO_MAP = dict()


# set the clang lib file path
def init_clang_lib_path():
    if not Config.loaded:
        # 或指定目录：Config.set_library_path("/usr/lib/x86_64-linux-gnu")
        Config.set_library_file(KitConfig.lib_clang_path())


def get_diag_info(diag):
    return {'info': diag.format(),
            'fixits': list(diag.fixits)}


def get_ref_def(cursor):
    """
    C++标准/内置的Node无referenced和definition，如UNEXPOSED_EXPR/XXX_LITERAL/XXX_OPERATOR/XXX_STMT，部分CALL_EXPR。

    声明的类/函数/变量等无referenced。
    方法/函数调用无definition，如CALL_EXPR。
    部分构造函数调用无definition。
    """
    ref = cursor.referenced
    decl = cursor.get_definition()
    result = list()
    if ref is not None:
        result.extend([f'ref: {ref.displayname}', f'{ref.location}'])  # {ref.extent.start.line}
    if decl is not None:
        result.extend([f'def: {decl.displayname}', f'{decl.location}'])
    return result


def cuda_enabled(file, include, namespace=None):
    """判断该文件是否为加速库内cuda相关文件。"""
    if not isinstance(include, list):
        include = [include]

    for x in include:
        if x == '':
            continue

        if x == 1 or x in file:
            return True
    return False


def usr_namespace(cursor, namespaces):
    """解析get_usr中的命名空间。

    例如：
    "c:@N@cv@ST>1#T@Ptr"：cv::Ptr<T>
    "c:@N@cv@E@WindowFlags@WINDOW_OPENGL"：cv::WindowFlags.WINDOW_OPENGL
    "c:@N@cv@N@cuda@S@GpuMat@F@GpuMat#*$@N@cv@N@cuda@S@GpuMat@S@Allocator#"：cv::cuda::CpuMat::GptMat
    "c:@N@cv@S@Ptr>#$@N@cv@N@cudacodec@S@VideoReader@F@operator->#1"：cv::Ptr<cv::cudacodec::VideoReader>
    """
    if not namespaces or not cursor.referenced:
        return ''
    if not isinstance(namespaces, list):
        namespaces = [namespaces]
    usr = cursor.referenced.get_usr()
    index = usr.find(cursor.referenced.spelling)  # 忽略"@S@GpuMat@F@GpuMat"这种重复的影响
    if index == -1:
        return ''
    nsc = re.findall(r'(?:@N@\w+){1,1000}', usr[:index])
    nss = ['::'.join(x[3:].split('@N@')) for x in nsc]
    for namespace in namespaces:
        for ns in nss:
            if namespace in ns:  # namespace可能是pattern，不是完整namespace
                return ns
    return ''


def in_acc_lib(file, cursor):
    """判断该文件是否为加速库文件。"""
    if not file:
        return False, False, ''
    file = file.replace("\\", os.path.sep).replace("/", os.path.sep)
    for lib, v in KitConfig.ACC_LIBS.items():
        if lib not in file:  # 待ACC_LIBS的Pattern改为全路径后，可以使用file.startswith(lib)
            continue
        if not v:
            cuda_en = False
            usr_ns = ''
        else:
            # get relative path
            new_file = file if not file.startswith(lib) else file.replace(lib, '')
            cuda_en = cuda_enabled(new_file, v.cuda_include)
            usr_ns = usr_namespace(cursor, v.namespace)
            cursor.lib = v.lib_name
        return True, cuda_en, usr_ns
    return False, False, ''


def find_right_angle(api):
    """找到右尖括号的位置"""
    stack = 0
    r = -1
    for i, c in enumerate(api):
        if c == '<':
            stack += 1
        elif c == '>':
            stack -= 1
            if stack == 0:
                r = i
                break
    return r


def filter_acc(cursor):
    hit = False
    result_type = None
    spelling = None
    api = None
    definition = None
    source = None
    cuda_en = False
    ns = ''

    if cursor.kind.name in helper_dict:  # 用于提前对节点进行处理，比如VAR_DECL的命名空间、FUNCTIONPROTO的参数等
        result_type, spelling, api, definition, source = helper_dict[cursor.kind.name](cursor)
    if cursor.kind.name in filter_dict:
        result_type, spelling, api, definition, source = filter_dict[cursor.kind.name](cursor)
        hit, cuda_en, ns = in_acc_lib(source, cursor)

    # 从get_usr中解析的namespace（连续的(?:@N@\w+)），Cursor解析得到的API，前者更完整。
    # 用户代码dnn::Net，get_user得到cv::dnn::dnn4_v20211220，Cursor得到dnn::Net，取cv::dnn::Net
    if ns and api:
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
    # 某些情况下，如用户代码中使用`using namespace cv; using namespace dnn;`，导致api解析出来的namespace也带有dnn4_v20211004，与
    # usr_namespace提取的namespace作用后也无法消除dnn4_v20211004，需特殊处理
    if api and api.startswith('cv::'):
        ns_api = api.split('::')
        namespaces, base_api = ns_api[:-1], ns_api[-1:]
        rm_idx = None
        for i, name in enumerate(namespaces):
            pattern = r".+_v\d+"
            if re.match(pattern, name):
                rm_idx = i

        if rm_idx is not None:
            namespaces.pop(rm_idx)
            api = '::'.join(namespaces + base_api)
    return hit, Info(result_type, spelling, api, definition, source), cuda_en


def get_includes(tu):
    """
    获取FileInclusion，没检测到/不存在对应库文件时，不生效。如：
        #include <fstream>
        #include <opencv2>
        #include <person.hpp>
    """
    includes = tu.get_includes()
    logger.debug('depth, include.name, location.line, location.column, source')
    srcs = list()
    for x in includes:
        if x.depth < 2:  # 1,2,3...
            logger.debug(x.depth, x.include.name, get_attr(x, 'location.line'),
                         get_attr(x, 'location.column'), x.source)
            srcs.append(x.include.name)
    return srcs


def macro_map(cursor, file=None):
    """过滤并保存宏定义到字典中，主要用于标识符重命名场景。

    如：#define cublasCreate         cublasCreate_v2

    TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD需打开。
    """
    if cursor.kind == CursorKind.MACRO_DEFINITION:
        if not file or not is_acc_path(file):
            return

        tk = list(cursor.get_tokens())
        if len(tk) == 2 and not tk[0].spelling.startswith('_') and tk[1].kind.name == 'IDENTIFIER':
            MACRO_MAP[tk[1].spelling] = tk[0].spelling


def typedef_map(cursor, file):
    """过滤并保存typedef到字典中，主要用于标识符重命名场景。

    typedef Rect_<int> Rect2i; typedef Rect2i Rect;
    c.type.get_canonical(): cv::Rect_<int>
    c.type.get_typedef_name(): Rect2i
    """
    if not file or not is_acc_path(file):
        return
    if cursor.kind == CursorKind.TYPEDEF_DECL:
        # 通过get_canonical()可以获取最原始类型，但underlying_typedef_type获取的是当前声明对应的源类型。
        TYPEDEF_MAP[cursor.type.get_canonical().spelling] = cursor.type.get_typedef_name()
    return


def actual_arg(cursor):
    """获取调用时传递的实参，忽略隐式类型转换/实例化。Cursor.kind应为CALL_EXPR。
    """
    for _ in range(KitConfig.CURSOR_DEPTH):
        # CursorKind: MEMBER_REF_EXPR, DECL_REF_EXPR, TYPE_REF, STRING_LITERAL, INTEGER_LITERAL ...
        if 'REF' in cursor.kind.name or 'LITERAL' in cursor.kind.name:
            break
        children = get_children(cursor)
        if children:
            cursor = children[0]
    return cursor


def parent_stmt(cursor):
    """获取所属Statement对应的代码"""
    for _ in range(KitConfig.CURSOR_DEPTH):
        # CursorKind: DECL_STMT, DEFAULT_STMT ...
        if cursor.kind.name.endswith('STMT'):
            break
        parent = get_attr(cursor, 'parent')
        if parent:
            cursor = parent
    return cursor


def parse_args(node):
    args = list()
    if node.kind == CursorKind.CALL_EXPR:
        refs = node.referenced
        if not refs:
            return args

        parameters = [f'{x.type.spelling} {x.spelling}' for x in node.referenced.get_arguments()]
        arguments = list(node.get_arguments())
        # 构造函数调用时，get_arguments()获取不到实参，referenced.get_arguments()可以获取形参。
        if parameters and not arguments:
            ref_end = -1
            # 隐式调用（通常为构造函数调用）子节点均为参数，显式(构造函数)调用子节点包含命名空间+（类型）引用+参数。
            children = get_children(node)
            for i, ci in enumerate(children):
                if ci.kind not in [CursorKind.NAMESPACE_REF, CursorKind.TEMPLATE_REF, CursorKind.TYPE_REF]:
                    break
                elif ci.kind == CursorKind.TYPE_REF:
                    ref_end = i
                    break
            arguments = children[ref_end + 1:]

        for param, x in zip(parameters, arguments):
            x = skip_implicit(x)
            if not x:  # 有默认值的Keyword参数，如果实参未传，则为None
                continue
            x = actual_arg(x)
            # 或直接读取代码：read_cursor(x)
            spelling = auto_match(x).spelling  # 参数通常未记录info，无法获取info.spelling
            if is_user_code(get_attr(x, 'referenced.location.file.name')):
                start = get_attr(x, 'referenced.extent.start')
                src_loc = f"{get_attr(start, 'file.name')}, {get_attr(start, 'line')}:" \
                          f"{get_attr(start, 'column')}"
                src_code = read_cursor(x.referenced)
            else:
                src_loc = 'NO_REF'
                src_code = 'NO_REF'
            args.append(f'{param} | {spelling} | {src_code} | {src_loc}')
    return args


def node_debug_string(node, children):
    location = f"{get_attr(node, 'extent.start.file.name')}, {get_attr(node, 'extent.start.line')}:" \
               f"{get_attr(node, 'extent.start.column')}-{get_attr(node, 'extent.end.column')}"

    # node的属性和方法：kind.name/type.kind.name/get_usr()/displayname/spelling/type.spelling/hash
    # 其他可记录信息：get_attr(node, 'referenced.kind.name')/api/result_type/source/definition/get_ref_def(node)/children
    info = {
        'kind': node.kind.name,
        'type_kind': node.type.kind.name,
        'ref_kind': get_attr(node, 'referenced.kind.name'),
        'spelling': node.spelling,
        'type': node.type.spelling,
        'location': location,
        'children': children
    }

    return info


def parse_info(node, cwd=None):
    if node.kind == CursorKind.TRANSLATION_UNIT:
        file = node.spelling
    else:
        if not get_attr(node, 'location.file'):
            file = None
        else:
            file = os.path.normpath(node.location.file.name)

    macro_map(node, file)
    typedef_map(node, file)

    # 如果对于系统库直接返回None，可能会导致部分类型无法解析，但是解析系统库会导致性能下降。
    usr_code = is_user_code(file)

    if usr_code:
        SCANNED_FILES.append(file)
        if not getattr(node, 'scanned', False) and not getattr(node, 'implicit', False):
            hit, (result_type, spelling, api, definition, source), cuda_en = filter_acc(node)
        else:
            hit = False

        if hit:
            api = MACRO_MAP.get(api, api)
            loc = f"{get_attr(node, 'extent.start.file.name')}, {get_attr(node, 'extent.start.line')}:" \
                  f"{get_attr(node, 'extent.start.column')}"
            args = parse_args(node)
            item = {
                KitConfig.ACC_API: api,
                KitConfig.CUDA_EN: cuda_en,
                KitConfig.LOCATION: loc,
                KitConfig.CONTEXT: args,
                KitConfig.ACC_LIB: get_attr(node, 'lib'),
            }
            RESULTS.append(item)

    children = list()
    for c in get_children(node):
        c_info = parse_info(c, cwd)
        if c_info:
            children.append(c_info)

    info = None
    if usr_code:
        info = node_debug_string(node, children)

    return info


class Parser:
    # creates the object, does the inital parse
    def __init__(self, path):
        # delay init clang lib file path to here
        init_clang_lib_path()

        logger.info(f'Scanning file: {path}')
        self.index = Index.create()  # 若为单例模型，是否有加速作用
        # args: '-Xclang', '-ast-dump', '-fsyntax-only', '-std=c++17', "-I/path/to/include"
        # option: TranslationUnit.PARSE_PRECOMPILED_PREAMBLE, TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        args = [f'-I{x}' for x in KitConfig.INCLUDES.values() if x]
        if platform.system() == "Windows":
            args.append("--target=x86_64-w64-windows-gnu")
        if KitConfig.CXX_STD:
            args.append(f'-std={KitConfig.CXX_STD}')
        self.tu = self.index.parse(path,
                                   args=args,
                                   options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)

    def parse(self):
        global RESULTS, SCANNED_FILES, MACRO_MAP
        RESULTS.clear()
        MACRO_MAP.clear()
        TYPEDEF_MAP.clear()

        for d in self.tu.diagnostics:
            logger.warning(f'Code diagnose：{get_diag_info(d)}')
            if d.severity > KitConfig.TOLERANCE:
                logger.warning(f'Diagnostic severity {d.severity} > tolerance {KitConfig.TOLERANCE}, skip this file.')
                return dict()

        cwd = os.path.dirname(self.tu.spelling)  # os.path.abspath(os.path.normpath(tu.spelling))
        start = time.time()
        info = parse_info(self.tu.cursor, cwd=cwd)
        logger.debug(f'Time elapsed： {time.time() - start:.3f}s')
        if logger.level == logging.DEBUG:
            dump = self.tu.spelling.replace('/', '.')
            os.makedirs('temp', exist_ok=True)
            temp_json_file = os.path.join('temp', f'{dump}.json')
            IOUtil.json_safe_dump(info, temp_json_file)
            logger.debug(f'Ast saved in：{temp_json_file}')

        return deepcopy(RESULTS)


if __name__ == '__main__':
    p = Parser('../examples/classify.cpp')
    p.parse()
