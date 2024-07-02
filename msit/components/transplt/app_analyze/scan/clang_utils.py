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

"""
## type/referenced
结论：函数/方法使用referenced.result_type做返回类型，其他使用type做返回类型。

函数/方法/构造函数定义XXX_DECL（referenced）的type包含参数type。
仅函数/方法/构造函数定义XXX_DECL（referenced）的result_type非空。
定义本身无referenced。
构造函数定义XXX_DECL（referenced）返回void。
函数/方法的type为返回类型。构造函数的type为类的类型。

函数/方法调用（CALL_EXPR）：referenced为函数/方法定义
- type不包含参数type，result_type为空。
- referenced.type包含参数type，referenced.result_type不会包含参数type。
- type和referenced.result_type基本相同，但后者更准确，如对于basic_string，前者为xxx, 后者为xxx &

类调用/构造函数调用（CALL_EXPR）：referenced为构造函数
- type不包含参数type，result_type为空。
- referenced.type为void包含参数type，referenced.result_type为void。

函数/方法引用（MEMBER_REF_EXPR/DECL_REF_EXPR）：referenced为函数/方法定义
- type会包含参数type，result_type为空。
- referenced.type包含参数type，referenced.result_type不包含参数type。
- 与函数/方法调用（CALL_EXPR）相同。

类/属性/变量引用（TYPE_REF/MEMBER_REF_EXPR/DECL_REF_EXPR）：referenced为定义
- type同referenced.type为类型。result_type与referenced.result_type为空。

参数/变量声明（PARM_DECL/VAR_DECL）：referenced为所属类型的定义
- type同referenced.type为类型。result_type与referenced.result_type为空。

命名空间（NAMESPACE_REF）：referenced为定义
- 全为空。

字面量（XXX_LITERAL）：无referenced
- type为类型，result_type为空。无referenced.type，referenced.result_type。

运算符（XXX_OPERATOR）：无referenced
- type为返回类型，result_type为空。无referenced.type，referenced.result_type。

数组取下表（ARRAY_SUBSCRIPT_EXPR）：无referenced
- type为返回元素的类型，result_type为空。无referenced.type，referenced.result_type。
- 同运算符。

XXX_STMT：无referenced
- type、result_type为空。无referenced.type，referenced.result_type。

TRANSLATION_UNIT：无referenced
- type、result_type为空。无referenced.type，referenced.result_type。


## definition

C++标准/内置的Node：无referenced和definition
- 如UNEXPOSED_EXPR/XXX_LITERAL/XXX_OPERATOR/XXX_STMT/PAREN_EXPR，部分CALL_EXPR。

方法/函数调用（CALL_EXPR）：无definition

部分构造函数调用：无definition

TRANSLATION_UNIT：无definition

INCLUSION_DIRECTIVE：无definition
"""

import re
from collections import namedtuple

from clang.cindex import CursorKind, TypeKind

from app_analyze.common.kit_config import KitConfig
from app_analyze.utils.lib_util import get_sys_path

SYS_PATH = get_sys_path()
TYPEDEF_MAP = dict()

Info = namedtuple('Info', ['result_type', 'spelling', 'api', 'definition', 'source'])


def is_user_code(file):
    return file and all(not file.startswith(p) for p in SYS_PATH)


def get_attr(obj, attr=None, default_val=None):
    """"""
    attrs = attr.split('.')
    for a in attrs:
        obj = getattr(obj, a, default_val)
        if not obj:
            return obj
    return obj


def group_namespace(cursor):
    """此处将template_ref和type_ref分开处理，如cv::Ptr<cv:cuda::VideoReader>分为cv::Ptr和cv:cuda::VideoReader"""
    cursors = get_children(cursor.parent)
    groups = [list()]
    group = [cursor]
    for c in cursors:
        if c.kind == CursorKind.NAMESPACE_REF:
            groups[-1].append(c)
            if cursor == c:
                group = groups[-1]
        elif c.kind in [CursorKind.TEMPLATE_REF, CursorKind.TYPE_REF]:
            groups[-1].append(c)
            groups.append(list())
    return group


def merge_namespace(ns, api):
    """合并namespace和API，去除中间namespace。

    从连续的NAMESPACE_REF节点组合而成的namespace，TYPE_REF节点的type.spelling，后者更完整。
    用户代码cv::dnn::Net，解析的是cv::dnn::Net或cv::dnn::dnn4_v20211220::Net（两者等价），尽量取前者。
    """
    if not ns or not api:
        return api
    ns_end = api.rfind('::')
    if ns_end == -1:  # api无命名空间
        nsapi = f'{ns}{api}'
    else:
        api_ns = api[:ns_end + 2]
        api_core = api[ns_end + 2:]
        ns_idx = api_ns.find(ns)
        if not api_ns.startswith(ns) and ns_idx != -1:  # x.startswith('')为True
            nsapi = f'{api_ns[:ns_idx]}{ns}{api_core}'
        else:
            nsapi = f'{ns}{api_core}'
    return nsapi


def get_namespace(children, suffix=True):
    """例如cv::cudacodec::VideoReader

    特殊示例：OpenCV FileStorage::READ，DECL_REF_EXPR，将子节点TYPE_REF作为namespace处理。
    {"kind": "DECL_REF_EXPR","type_kind": "ENUM","spelling": "READ","type": "cv::FileStorage::Mode"}
        {"kind": "TYPE_REF","type_kind": "RECORD","spelling": "class cv::FileStorage","type": "cv::FileStorage"}

    特殊示例：OpenCV cv::cuda::HOG::create()，静态方法，CALL_EXPR->DECL_REF_EXPR，将子节点TYPE_REF作为namespace处理。
    {"kind": "CALL_EXPR","type_kind": "UNEXPOSED","spelling": "create","type": "Ptr<cv::cuda::HOG>"}
        {"kind": "DECL_REF_EXPR","type_kind": "FUNCTIONPROTO","spelling": "create","type": "Ptr<cv::cuda::HOG> (...)"}
    """
    namespace = ''
    tpl_flag = list()  # 使用栈结构保存template的Flag，应对嵌套template场景
    for child in children:
        if child.kind == CursorKind.NAMESPACE_REF:
            child.scanned = True
            namespace += child.spelling + '::'
        elif child.kind == CursorKind.TEMPLATE_REF:
            child.scanned = True
            namespace += child.spelling + '<'
            tpl_flag.append(True)
        elif child.kind == CursorKind.TYPE_REF:  # 特殊示例
            child.scanned = True
            ct = child.type.spelling
            type_core = ct.split('::')[-1]
            if not ct.startswith(namespace) and namespace in ct:  # 理论上前者满足时，后者必须满足，无需多加判断
                # namespace为部分，例如代码cuda::GpuMat，namespace为cuda，type_ref为cv::cuda::GpuMat。
                # 特殊代码dnn::Net，namespace为dnn，type_ref为cv::dnn::dnn4_v20211220::Net，希望为cv::dnn::Net
                namespace = f'{ct[:ct.find(namespace)]}{namespace}{type_core}::'
            else:
                namespace = f"{namespace}{type_core}{' > ' if tpl_flag and tpl_flag.pop() else '::'}"
        else:  # 应该不会出现，用作保险
            break
    namespace = namespace.strip('<')  # 防止遗留<
    return namespace if suffix else namespace.strip(':')  # 空字符串不会报错


def get_typedef(canonical_type):
    for _ in range(10):
        if canonical_type in TYPEDEF_MAP:
            canonical_type = TYPEDEF_MAP.get(canonical_type)
        else:
            break
    return canonical_type


def off_alias(t):
    """例如using T = cv::cuda::GpuMat;"""
    type_decl = t.get_declaration()
    if not type_decl:  # 不确定会否出现
        return t
    if not is_user_code(get_attr(type_decl, 'location.file.name')):
        return t

    # 其他情况待确认：NAMESPACE_ALIAS，TYPE_ALIAS_TEMPLATE_DECL
    true_type = t
    if type_decl.kind == CursorKind.TYPE_ALIAS_DECL:
        # 通过get_canonical()可以获取最原始类型，但underlying_typedef_type获取的是当前声明对应的源类型。
        true_type = type_decl.underlying_typedef_type
    return true_type


def get_children(cursor):
    if not cursor:
        return list()
    if not hasattr(cursor, 'children'):
        cursor.children = list(cursor.get_children())  # 临时属性，便于特殊处理子节点
    for child in cursor.children:
        child.parent = cursor
        true_type = off_alias(child.type)
        if child.type != true_type:
            child._type = true_type
    return cursor.children


def set_attr_children(cursor, attr, value):
    for child in get_children(cursor):
        setattr(child, attr, value)


def skip_implicit(cursor):
    """通常是隐式类型转换或者指针。

    一个隐式转换应只有1个后继节点。
    一个指针应只有1个后继节点，c.type.kind == TypeKind.POINTER
    """
    if cursor.kind == CursorKind.UNEXPOSED_EXPR:
        children = get_children(cursor)
        if children:
            return skip_implicit(children[0])
        else:
            ret = None
            return ret
    return cursor


def strip_implicit(type_str):
    """消去CALL_EXPR result_type中的const或指针符号。"""
    if not type_str:
        return type_str
    type_str = type_str.replace('const ', '')  # 去除const
    type_str = type_str.strip(' *')  # 去除末尾指针符号*
    return type_str


def read_code(file_path, start, end):
    with open(file_path, 'rb') as f:
        f.seek(start)
        return f.read(end - start)


def read_cursor(cursor):
    file = get_attr(cursor, 'location.file.name')
    start = get_attr(cursor, 'extent.start.offset')
    end = get_attr(cursor, 'extent.end.offset')
    spelling = None
    if file and start and end:
        spelling = read_code(file, start, end)
    return spelling


##### Cursor解析函数（开始），每个函数匹配一个/一类CursorKind
def default(c):
    """
    特殊示例：
    OpenCV代码：types.hpp
        typedef Rect_<int> Rect2i; typedef Rect2i Rect;
    用户代码：cascadeclassifier.cpp
        vector < Rect > faces;
        faces[i];

    c.type: __gnu_cxx::__alloc_traits<std::allocator<cv::Rect_<int> > >::value_type
    c.reference.result_type: std::vector<cv::Rect_<int>, std::allocator<cv::Rect_<int> > >::reference
    c.type.get_canonical(): cv::Rect_<int>
    c.type.get_typedef_name(): value_type

    Returns:
        Info
    """
    if c.referenced:
        definition = c.referenced.displayname
        source = get_attr(c.referenced, 'location.file.name')
    else:
        definition = c.displayname
        source = get_attr(c, 'location.file.name')
    # C++ typedef如: typedef Size_<int> Size2i; typedef Size2i Size;
    # 对应的：c.spelling = Size_，c.type.spelling = cv::Size
    if re.search(rf'\b{re.escape(c.spelling)}\b', c.type.get_canonical().spelling):
        api = c.type.spelling
        api = strip_implicit(api)
    else:
        api = c.spelling

    result_type = c.type.spelling
    if 'std::' in result_type:  # 针对容器类型索引后的返回类型做处理，判断需要更准确
        if c.type.get_canonical():
            # get_canonical获取的是原始类型，需映射为typedef的最终类型
            result_type = get_typedef(c.type.get_canonical().spelling)
    c.info = Info(result_type, c.spelling, api, definition, source)
    return c.info


def var_decl(c):
    """变量声明，不包括赋值。

    通常TypeKind为：TYPEDEF，RECORD
    extent包含类型和变量名，无需修改。
    如果非原生类型，通常后继节点有命名空间NAMESPACE_REF（可选），类型引用TYPE_REF，构造函数CALL_EXPR（可选），可通过referenced获取源头。
    示例1：vector<vector<cv::Point> >

    Returns:
        Info
    """
    children = get_children(c)
    result_type, spelling, api, definition, source = default(c)
    if children:  # TYPEDEF RECORD
        # 此处将template_ref和type_ref合并处理，如cv::Ptr<cv:cuda::VideoReader>
        spelling = get_namespace(children) + spelling  # TYPE VAR
        for child in children:
            # 原生类型变量声明 = 加速库/非加速库调用，则没有TYPE_REF
            if child.kind == CursorKind.TYPE_REF:  # 从变量声明所属类型获取source
                definition = get_attr(child, 'referenced.displayname')
                source = get_attr(child, 'referenced.location.file.name')
                break

    c.info = Info(result_type, spelling, api, definition, source)
    return c.info


def parm_decl(c):
    """参数声明，不包括赋值。

    extent包含类型和变量名，无需修改。
    如果非原生类型，后继节点有命名空间NAMESPACE_REF（可选），类型引用TYPE_REF，可通过referenced获取源头。

    Returns:
        Info
    """
    children = get_children(c)
    result_type, spelling, api, definition, source = default(c)
    if children:
        # 此处将template_ref和type_ref合并处理，如cv::Ptr<cv:cuda::VideoReader>
        spelling = get_namespace(children) + spelling  # TYPE VAR
        for child in children:  # 示例2
            # 原生类型变量声明 = 加速库/非加速库调用，则没有TYPE_REF
            if child.kind == CursorKind.TYPE_REF:  # 从变量声明所属类型获取source
                definition = get_attr(child, 'referenced.displayname')
                source = get_attr(child, 'referenced.location.file.name')
                break
    if c.type.kind in (TypeKind.RVALUEREFERENCE, TypeKind.LVALUEREFERENCE):
        api = c.type.get_pointee().spelling
    else:
        api = c.type.spelling
    c.info = Info(result_type, spelling, api, definition, source)
    return c.info


def call_expr(c):
    """
    函数调用：a()
    1. CALL_EXPR：a
        1. 函数引用：DECL_REF_EXPR：a
        2. 参数

    方法调用：a.b()
    1. CALL_EXPR：b
        1. 方法引用：MEMBER_REF_EXPR：b
            1. 对象引用：DECL_REF_EXPR：a
        2. 参数

    重载运算：operator=, a=b
    1. CALL_EXPR：operator=
        1. 对象引用：DECL_REF_EXPR：a
        2. 函数引用：DECL_REF_EXPR：operator=
        3. 参数

    类调用：cv::Size()
    1. CALL_EXPR：Size_，模板定义cv::Size
        1. 命名空间引用：NAMESPACE_REF：cv
        2. 类型引用：TYPE_REF：cv::Size
        3. 参数

    重载运算：std::cout << "..." << "..." << std::endl;
    1. CALL_EXPR：operator<<
        1. CALL_EXPR：operator<<
            1. CALL_EXPR：operator<<
                1. DECL_REF_EXPR of TYPEDEF：cout
                2. DECL_REF_EXPR of FUNCTIONPROTO: operator<<
                3. 参数
            2. DECL_REF_EXPR of FUNCTIONPROTO: operator<<，将1的结果作为类型引用
            3. 参数
        2. DECL_REF_EXPR of FUNCTIONPROTO: operator<<，将1的结果作为类型引用
        3. DECL_REF_EXPR of FUNCTIONPROTO: endl，将2的结果作为类型引用

    重载运算：cv::Size imgSize = img.size();
    1. CALL_EXPR：operator()
        1. MEMBER_REF_EXPR of RECORD：size
            1. DECL_REF_EXPR of ELABORATED: img
        2. DECL_REF_EXPR of FUNCTIONPROTO: operator()

    注意存在隐式类调用cv::Mat m，此时type为cv::Mat，definition为Mat，没有后继节点。
    extent包含命名空间、函数名、()，无需修改。
    通常CALL_EXPR可通过referenced获取源头，但示例1不适用。

    Returns:
        Info
    """
    children = get_children(c)
    if not children:
        return default(c)
    c0 = skip_implicit(children[0])
    if not c0:
        return default(c)

    result_type, spelling, api, definition, source = default(c)
    op_overload = 'operator' in c.spelling  # 需要判断的更加准确
    if op_overload:
        op = c.spelling  # {"spelling": "operator()"} 对应接口为`operator()`
        for i, child in enumerate(children):
            child = skip_implicit(child)
            if not child:
                continue
            if c.spelling == child.spelling and i > 0:  # 重载的运算符节点
                child.scanned = True  # 用于标识是否已被扫描
        cls, _, attr, _, _ = auto_match(c0)
        cls = strip_implicit(cls)  # 可能返回const xxx类型，去除const和末尾指针符号*
        if c0.spelling == c.spelling and get_attr(c0, 'referenced.kind') == get_attr(c, 'referenced.kind'):  # 运算符节点
            api = attr
            c0.scanned = True  # 用于标识是否已被扫描
        elif c0.kind == CursorKind.MEMBER_REF_EXPR:
            # 运算符重载，且最近子节点为MEMBER_REF_EXPR，则必为属性引用，而非方法引用，否则最近子节点为CALL_EXPR。
            api = f"{attr if op == '()' else cls}.{op}"  # 例如呈现cv::Mat.size()，实际为cv::MatSize()，待商榷
            c0.scanned = True
        else:
            api = f"{cls}.{op}"  # 例如呈现cv::FileStorage[]
            if c0.kind == CursorKind.DECL_REF_EXPR:
                c0.scanned = True
    else:
        # 1. c0 spelling、ref_kind和c一致，则c0为方法/函数引用，CursorKind为REF。
        if c0.spelling == c.spelling and get_attr(c0, 'referenced.kind') == get_attr(c, 'referenced.kind'):
            api = auto_match(c0).api
            if c0.kind == CursorKind.MEMBER_REF_EXPR:
                c0.scanned = True
            # 静态方法掉用，c.referenced.is_static_method() == True，c0为DECL_REF_EXPR
            elif c0.kind in [CursorKind.DECL_REF_EXPR, CursorKind.TYPE_REF]:
                c0.scanned = True
        # 2. 类/构造函数调用，如cv::Size(w, h)，不满足1。隐式调用子节点均为参数，显式调用子节点包含命名空间+（类型）引用+参数。
        # 如：{"kind": "CALL_EXPR","ref_kind": "CONSTRUCTOR","spelling": "Size_"}
        #        "kind": "TYPE_REF","ref_kind": "TYPEDEF_DECL","spelling": "cv::Size"
        elif get_attr(c, 'referenced.kind') == CursorKind.CONSTRUCTOR:
            # 获取TYPE_REF出现的位置，之前是命名空间等，之后是参数
            ref_end = -1
            for i, ci in enumerate(children):
                if ci.kind not in [CursorKind.NAMESPACE_REF, CursorKind.TEMPLATE_REF, CursorKind.TYPE_REF]:
                    break
                elif ci.kind == CursorKind.TYPE_REF:
                    ref_end = i
                    break
            for ci in children[:ref_end + 1]:
                ci.scanned = True
        elif get_attr(c, 'referenced.kind') == CursorKind.CXX_METHOD:
            if c.parent and get_attr(c, 'referenced.kind') == CursorKind.CLASS_DECL:
                api = c.parent.spelling + '::' + spelling
    c.info = Info(result_type, spelling, api, definition, source)
    return c.info


def member_ref_expr(c):
    """
    extent包含对象名、方法名，不包括()、参数，需修改。

    - 通常有后继节点：对象名DECL_REF_EXPR，函数名OVERLOADED_DECL_REF（可选），可通过referenced获取源头，但示例1不适用。
    - 若为类属性，则无后继节点。

    Returns:
        Info
    """
    api = c.spelling  # 示例1、7不适用
    if get_attr(c, 'referenced.kind') == CursorKind.FIELD_DECL:
        result_type = c.type.spelling
    else:  # CXX_METHOD
        result_type = get_attr(c, 'referenced.result_type.spelling')
        if get_attr(c, 'referenced.kind') == CursorKind.CONVERSION_FUNCTION:
            api = f'operator {result_type}'  # 例如示例7，从"operator unsigned long" 改为 "size_t"
    definition = get_attr(c, 'referenced.displayname')
    source = get_attr(c.referenced, 'location.file.name')
    children = get_children(c)
    if not children:
        return default(c)
    c0 = skip_implicit(children[0])
    if not c0:
        return default(c)
    if c0.kind == CursorKind.DECL_REF_EXPR:
        c0.scanned = True
        cls, obj, _, _, _ = decl_ref_expr(c0)  # 对象名的source在声明/实例化对象的地方，不在加速库里
    else:
        cls, obj = auto_match(c0)[:2]

    cls = strip_implicit(cls)  # 去除CALL_EXPR返回类型中的const和末尾指针符号*
    c.info = Info(result_type, f'{obj}.{api}', f'{cls}.{api}', definition, source)
    return c.info


def decl_ref_expr(c):
    """
    Returns:
        Info
    """
    # 若为函数引用，需从children[0]的namespace获取
    # extent包含对象名、方法名，不包括()、参数，需修改。
    # type为<overloaded function type>，referenced
    definition = c.referenced.displayname if c.referenced else None
    children = get_children(c)
    # get_namespace会设置scanned属性，set_attr_children(c, 'scanned', True)
    namespace = get_namespace(children)
    if c.type.kind == TypeKind.OVERLOAD:  # 似乎没再遇到过
        if len(children) < 1:
            raise RuntimeError(f'DECL_REF_EXPR of Typekind OVERLOAD {c.spelling} {c.location} 应有后继节点：' \
                               f'命名空间NAMESPACE_REF（可选） 函数名OVERLOADED_DECL_REF')
        api = spelling = namespace + children[-1].spelling
        result_type = get_attr(children[-1], 'referenced.result_type.spelling')  # 函数返回值类型
        source = get_attr(children[0], 'referenced.location.file.name')
    # extent包含对象名、方法名，不包括()、参数，需修改。
    elif c.type.kind == TypeKind.FUNCTIONPROTO:
        # 静态方法引用，c.referenced.is_static_method() == True，有后继节点：类型TYPE_REF
        api = spelling = namespace + c.spelling
        result_type = get_attr(c, 'referenced.result_type.spelling')  # 函数返回值类型
        source = get_attr(c, 'referenced.location.file.name')
    # 若为对象引用，可从type获取类型归属，ELABORATED/RECORD/CONSTANTARRAY/TYPEDEF/ENUM
    # extent包含对象名，无需修改。
    else:
        result_type, spelling, api, definition, source = default(c)
        spelling = namespace + spelling
        if c.type.kind == TypeKind.ENUM:
            if 'anonymous enum' not in c.type.spelling:
                api = f'{c.type.spelling}.{api}'
                # 例如'cv::dnn::', 'cv::dnn::dnn4_v20211220::Backend.DNN_BACKEND_CUDA'
                api = merge_namespace(namespace, api)
    c.info = Info(result_type, spelling, api, definition, source)
    return c.info


def overloaded_decl_ref(c):
    return default(c)


def array_subscript_expr(c):
    """对应TypeKind.CONSTANTARRAY，运算符[]重载不会被当做此类型。

    1. ARRAY_SUBSCRIPT_EXPR：数组索引表达式
        1. 嵌套对象/DECL_REF_EXPR/MEMBER_REF_EXPR
        2. 索引

    Args:
        c: Cursor

    Returns:
        Info
    """
    return default(c)


def template_ref(c):
    return default(c)


def type_ref(c):
    """
    Args:
        c: cursor.

    Returns:
        Info
    """
    definition = get_attr(c, 'referenced.displayname')
    source = get_attr(c, 'referenced.location.file.name')
    c.info = Info(c.type.spelling, c.spelling, c.type.spelling, definition, source)
    return c.info


def namespace_ref(c):
    """
    Args:
        c: cursor.

    Returns:
        Info
    """
    # 将连续的namespace合并
    group = group_namespace(c)
    api = get_namespace(group, suffix=False)
    # 某些情况下，命名空间cv的referenced.location为opencv2/cudaimgproc.hpp，被判断为CUDAEnable，故使用group[-1]（如TYPE_REF）的
    # referenced.location作为location。比如AIFaceLib/faceSubstitution/FaceSubstitutionOpencvDnnImpi.h, 159:9
    c.info = Info(None, c.spelling, api, None, get_attr(group[-1], 'referenced.location.file.name'))
    return c.info


def inclusion_directive(c):
    try:
        file = c.get_included_file()
    except AssertionError as e:
        c.info = Info(None, c.spelling, c.spelling, None, None)
    else:
        c.info = Info(None, c.spelling, c.spelling, None, file.name)

    return c.info


def literal(c):
    """XXX_LITERAL"""
    if c.kind == CursorKind.STRING_LITERAL:
        spelling = c.spelling
    else:
        spelling = read_cursor(c)
    c.info = Info(c.type.spelling, spelling, None, None, None)
    return c.info


def operator(c):
    """UNARY_OPERATOR/BINARY_OPERATOR/..."""
    spelling = read_cursor(c)
    c.info = Info(c.type.spelling, spelling, None, None, None)
    return c.info


##### Cursor解析函数（结束）
# 小粒度分析CursorKind
small_dict = {
    'MEMBER_REF_EXPR': member_ref_expr,
    'DECL_REF_EXPR': decl_ref_expr,
    'OVERLOADED_DECL_REF': overloaded_decl_ref,
    'TYPE_REF': type_ref,
    'TEMPLATE_REF': template_ref,
    'NAMESPACE_REF': namespace_ref,
    'INCLUSION_DIRECTIVE': inclusion_directive,
    'CALL_EXPR': call_expr
}

# 大粒度分析CursorKind
large_dict = {
}

# 工具性分析CursorKind
other_dict = {
    'ARRAY_SUBSCRIPT_EXPR': array_subscript_expr
}

whole_dict = {**small_dict, **large_dict, **other_dict}

filter_dict = small_dict
helper_dict = large_dict  # 用于提前对节点进行处理，比如VAR_DECL的命名空间、FUNCTIONPROTO的参数等
if KitConfig.LEVEL == 'large':
    filter_dict = {**small_dict, **large_dict}
    helper_dict = dict()


def auto_match(c):
    if c.kind.name in whole_dict:
        return whole_dict.get(c.kind.name)(c)
    elif c.kind.name.endswith('LITERAL'):
        return literal(c)
    elif c.kind.name.endswith('OPERATOR'):
        return operator(c)
    else:
        return default(c)
