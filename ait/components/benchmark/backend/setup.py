# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
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
import re
import platform
import logging
import sys

from pybind11 import get_cmake_dir

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

STATIC_VERSION = "0.0.2"
PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")


def is_legal_path_length(path):
    if len(path) > 4096:
        logger.error(f"file total path length out of range (4096)")
        return False
    dirnames = path.split("/")
    for dirname in dirnames:
        if len(dirname) > 255:
            logger.error(f"file name length out of range (255)")
            return False
    return True


def is_match_path_white_list(path):
    if PATH_WHITE_LIST_REGEX.search(path):
        logger.error(f"path:{path} contains illegal char")
        return False
    return True


def is_legal_args_path_string(path):
    # only check path string
    if not path:
        return True
    if not is_legal_path_length(path):
        return False
    if not is_match_path_white_list(path):
        return False
    return True


# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

# Avoid a gcc warning below:
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid
# for C/ObjC but not for C++


class BuildExt(build_ext):
    def build_extensions(self):
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()


cann_base_path = None
cann_lib_path = None


def get_cann_path():
    global cann_base_path
    global cann_lib_path
    set_env_path = os.getenv("CANN_PATH", "")
    if not set_env_path:
        set_env_path = os.environ.get("ASCEND_TOOLKIT_HOME")
        if not set_env_path:
            set_env_path = "/usr/local/Ascend/ascend-toolkit/latest/"
        else:
            set_env_path = set_env_path.split(':')[0]
    else:
        set_env_path = set_env_path.split(':')[0]
    if not is_legal_args_path_string(set_env_path):
        raise TypeError(f"env CANN_PATH:{set_env_path} is illegal")
    atlas_nnae_path = "/usr/local/Ascend/nnae/latest/"
    atlas_toolkit_path = "/usr/local/Ascend/ascend-toolkit/latest/"
    hisi_fwk_path = "/usr/local/Ascend/"
    check_file_path = "runtime/lib64/stub/libascendcl.so"
    if os.path.exists(os.path.join(set_env_path, check_file_path)):
        cann_base_path = set_env_path
    elif os.path.exists(atlas_nnae_path + check_file_path):
        cann_base_path = atlas_nnae_path
    elif os.path.exists(atlas_toolkit_path + check_file_path):
        cann_base_path = atlas_toolkit_path
    elif os.path.exists(hisi_fwk_path + check_file_path):
        cann_base_path = hisi_fwk_path
    cann_lib_path = f'{cann_base_path}/runtime/lib64/stub/'

    if cann_base_path is None:
        if platform.machine() == "x86_64":
            check_file_path = "runtime/lib64/stub/x86_64/libascendcl.so"
        elif platform.machine() == "aarch64":
            check_file_path = "runtime/lib64/stub/aarch64/libascendcl.so"
        if os.path.exists(os.path.join(set_env_path, check_file_path)):
            cann_base_path = set_env_path
        elif os.path.exists(atlas_nnae_path + check_file_path):
            cann_base_path = atlas_nnae_path
        elif os.path.exists(atlas_toolkit_path + check_file_path):
            cann_base_path = atlas_toolkit_path
        elif os.path.exists(hisi_fwk_path + check_file_path):
            cann_base_path = hisi_fwk_path

        if cann_base_path is None:
            raise RuntimeError('error find no cann path')

        if platform.machine() == "x86_64":
            cann_lib_path = f'{cann_base_path}/runtime/lib64/stub/x86_64/'
        elif platform.machine() == "aarch64":
            cann_lib_path = f'{cann_base_path}/runtime/lib64/stub/aarch64/'

    logger.info("find cann path: %s", cann_base_path)


get_cann_path()

ext_modules = [
    Pybind11Extension(
        'aclruntime',
        sources=[
            'base/module/DeviceManager/DeviceManager.cpp',
            'base/module/ErrorCode/ErrorCode.cpp',
            'base/module/Log/Log.cpp',
            'base/module/MemoryHelper/MemoryHelper.cpp',
            'base/module/Tensor/TensorBase/TensorBase.cpp',
            'base/module/Tensor/TensorBuffer/TensorBuffer.cpp',
            'base/module/Tensor/TensorContext/TensorContext.cpp',
            'base/module/ModelInfer/model_process.cpp',
            'base/module/ModelInfer/utils.cpp',
            'base/module/ModelInfer/SessionOptions.cpp',
            'base/module/ModelInfer/ModelInferenceProcessor.cpp',
            'base/module/ModelInfer/DynamicAippConfig.cpp',
            'base/module/ModelInfer/cnpy.cpp',
            'base/module/ModelInfer/pipeline.cpp',
            'python/src/PyInterface/PyInterface.cpp',
            'python/src/PyTensor/PyTensor.cpp',
            'python/src/PyInferenceSession/PyInferenceSession.cpp',
        ],
        include_dirs=[
            'python/include/',
            'base/include/',
            'base/include/Base/ModelInfer/',
            f'{cann_base_path}/runtime/include',
        ],
        library_dirs=[
            cann_lib_path,
        ],
        extra_compile_args=['--std=c++11', '-g3', '-fstack-protector-all'],
        extra_link_args=['-Wl,-z,relro,-z,now', '-s'],
        libraries=['ascendcl', 'acl_dvpp', 'acl_cblas'],
        language='c++',
        define_macros=[('ENABLE_DVPP_INTERFACE', 1), ('COMPILE_PYTHON_MODULE', 1)],
    ),
]

setup(
    name="aclruntime",
    version=STATIC_VERSION,
    author="ais_bench",
    author_email="aclruntime",
    url="https://xxxxx",
    description="A test project using pybind11 and aclruntime",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.6",
)
