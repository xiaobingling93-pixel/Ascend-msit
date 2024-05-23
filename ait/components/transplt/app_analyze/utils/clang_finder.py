# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
import argparse
import os
import platform
import stat

from app_analyze.utils.log_util import logger
from components.utils.file_open_check import FileStat, OpenException, READ_FILE_NOT_PERMITTED_STAT


def _get_lib_clang_path_win():
    lib_clang_file_name = "libclang.dll"
    sys_paths = os.environ.get("Path")
    for sys_path in sys_paths.split(";"):
        lib_clang_candidate = os.path.join(sys_path, lib_clang_file_name)
        if os.path.exists(lib_clang_candidate) and os.access(lib_clang_candidate, os.R_OK):
            logger.debug(f'found libclang.dll at {lib_clang_candidate}.')
            return lib_clang_candidate

    logger.error('Unable to locate libclang.dll file, ait transplt will not working.')
    raise RuntimeError("Unable to locate libclang.dll file, ait transplt will not working.")


def _get_lib_clang_path_linux():
    import clang

    libclang_so_path = os.path.join(os.path.dirname(clang.__file__), "native", "libclang.so")
    if os.path.exists(libclang_so_path):
        return libclang_so_path
    else:
        logger.warning(f"libclang so: {libclang_so_path} not found, may meet error lately.")

    # default dirs
    candidate_lib_dirs = [
        "/lib", "/lib64", "/usr/lib", "/usr/lib64",
        "/usr/local/lib", "/usr/local/lib64",
        f'/usr/lib/{platform.machine()}-linux-gnu',
    ]

    # find clang in LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", None)
    if ld_library_path is not None:
        candidate_lib_dirs.extend(ld_library_path.split(":"))

    # find clang in paths extracted from /etc/ld.so.conf and /etc/ld.so.conf.d
    def extract_ld_lib_dirs(file):
        if not os.path.exists(file) or not os.access(file, os.R_OK):
            return

        with open(file, "r") as tmp_file:
            for line in tmp_file:
                if line.startswith("include"):
                    continue
                candidate_lib_dirs.append(line.strip())

    extract_ld_lib_dirs("/etc/ld.so.conf")

    for _, _, files in os.walk("/etc/ld.so.conf.d"):
        for f in files:
            if f.endswith("conf"):
                extract_ld_lib_dirs(os.path.join("/etc/ld.so.conf.d", f))

    clang_supported_version = ['14', '10', '8', '7', '6']  # support these clang versions so far
    clang_lib_name_patterns = [
        'libclang-xx.so', 'libclang-xx.0.so', 'libclang.so.xx', 'libclang-xx.so.1', 'libclang-xx.0.so.1'
    ]
    clang_lib_names = []
    for version in clang_supported_version:
        for name in clang_lib_name_patterns:
            clang_lib_names.append(name.replace("xx", version))

    for candidate_lib_dir in candidate_lib_dirs:
        for candidate_lib_name in clang_lib_names:
            candidate = os.path.join(candidate_lib_dir, candidate_lib_name)
            candidate = os.path.realpath(candidate)
            if not os.path.exists(candidate):
                continue

            try:
                file_stat = FileStat(candidate)
            except OpenException:
                logger.error(f"lib clang path:{candidate} is illegal. Please check.")
                continue

            # System so lib files can be soft links or belong to root user,
            # so we cannot use file_stat.is_basically_legal to check security of external so files.
            if file_stat.permission & READ_FILE_NOT_PERMITTED_STAT > 0:
                logger.error(f"The file {candidate} is group writable, "
                             "or is others writable, as import file(or directory), "
                             "permission should not be over 0o755(rwxr-xr-x)")
                continue

            if not os.access(candidate, os.R_OK) or file_stat.permission & stat.S_IRUSR == 0:
                logger.error(f"Current user doesn't have read permission to the file {candidate}, "
                             "as import file(or directory), permission should be at least 0o400(r--------) ")
                continue

            logger.debug(f'found libclang so file at {candidate}.')
            return candidate

    logger.error('Unable to locate libclang so file, ait transplt will not working.')
    raise RuntimeError("Unable to locate libclang so file, ait transplt will not working.")


def get_lib_clang_path():
    if platform.system() == "Windows":
        return _get_lib_clang_path_win()
    else:
        return _get_lib_clang_path_linux()
