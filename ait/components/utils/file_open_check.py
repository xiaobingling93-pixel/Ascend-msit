# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import stat
import re
import logging


MAX_SIZE_UNLIMITE = -1  # 不限制，必须显式表示不限制，读取必须传入
MAX_SIZE_LIMITE_CONFIG_FILE = 10 * 1024 * 1024  # 10M 普通配置文件，可以根据实际要求变更
MAX_SIZE_LIMITE_NORMAL_FILE = 4 * 1024 * 1024 * 1024  # 4G 普通模型文件，可以根据实际要求变更
MAX_SIZE_LIMITE_MODEL_FILE = 100 * 1024 * 1024 * 1024  # 100G 超大模型文件，需要确定能处理大文件，可以根据实际要求变更

PATH_WHITE_LIST_REGEX_WIN = re.compile(r"[^_:\\A-Za-z0-9/.-]")
PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")

PERMISSION_NORMAL = 0o640  # 普通文件
PERMISSION_KEY = 0o600  # 密钥文件
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH

SOLUTION_LEVEL = 35
SOLUTION_LEVEL_WIN = 45
logging.addLevelName(SOLUTION_LEVEL, "\033[1;32m" + "SOLUTION" + "\033[0m")  # green [SOLUTION]
logging.addLevelName(SOLUTION_LEVEL_WIN, "SOLUTION_WIN")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SOLUTION_BASE_LOC = '\"gitee repo: Ascend/ait, wikis: ait_security_error_log_solution, chapter:'
SOFT_LINK_SUB_CHAPTER = 'soft_link_error_log_solution\"'
PATH_LENGTH_SUB_CHAPTER = 'path_length_overflow_error_log_solution\"'
OWNER_SUB_CHAPTER = 'owner_or_ownergroup_error_log_solution\"'
PERMISSION_SUB_CHAPTER = 'path_permission_error_log_solution\"'
ILLEGAL_CHAR_SUB_CHAPTER = 'path_contain_illegal_char_error_log_solution\"'


def solution_log(content):
    logger.log(SOLUTION_LEVEL, f"visit \033[1;32m {content} \033[0m for detailed solution")  # green content


def solution_log_win(content):
    logger.log(SOLUTION_LEVEL_WIN, f"visit {content} for detailed solution")


def is_legal_path_length(path):
    if len(path) > 4096 and not sys.platform.startswith("win"):  # linux total path length limit
        logger.error(f"file total path{path} length out of range (4096), please check the file(or directory) path")
        solution_log(SOLUTION_BASE_LOC + PATH_LENGTH_SUB_CHAPTER)
        return False

    if len(path) > 260 and sys.platform.startswith("win"):  # windows total path length limit
        logger.error(f"file total path{path} length out of range (260), please check the file(or directory) path")
        solution_log_win(SOLUTION_BASE_LOC + PATH_LENGTH_SUB_CHAPTER)
        return False

    dirnames = path.split("/")
    for dirname in dirnames:
        if len(dirname) > 255:  # linux single file path length limit
            logger.error(f"file name{dirname} length out of range (255), please check the file(or directory) path")
            solution_log(SOLUTION_BASE_LOC + PATH_LENGTH_SUB_CHAPTER)
            return False
    return True


def is_match_path_white_list(path):
    if PATH_WHITE_LIST_REGEX.search(path) and not sys.platform.startswith("win"):
        logger.error(f"path:{path} contains illegal char, legal chars include A-Z a-z 0-9 _ - / .")
        solution_log(SOLUTION_BASE_LOC + ILLEGAL_CHAR_SUB_CHAPTER)
        return False
    if PATH_WHITE_LIST_REGEX_WIN.search(path) and sys.platform.startswith("win"):
        logger.error(f"path:{path} contains illegal char, legal chars include A-Z a-z 0-9 _ - / . : \\")
        solution_log_win(SOLUTION_BASE_LOC + ILLEGAL_CHAR_SUB_CHAPTER)
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


class OpenException(Exception):
    pass


class FileStat:
    def __init__(self, file) -> None:
        if not is_legal_path_length(file) or not is_match_path_white_list(file):
            raise OpenException(f"create FileStat failed")
        self.file = file
        self.is_file_exist = os.path.exists(file)
        if self.is_file_exist:
            self.file_stat = os.stat(file)
            self.realpath = os.path.realpath(file)
        else:
            self.file_stat = None

    @property
    def is_exists(self):
        return self.is_file_exist

    @property
    def is_softlink(self):
        return os.path.islink(self.file) if self.file_stat else False

    @property
    def is_file(self):
        return stat.S_ISREG(self.file_stat.st_mode) if self.file_stat else False

    @property
    def is_dir(self):
        return stat.S_ISDIR(self.file_stat.st_mode) if self.file_stat else False

    @property
    def file_size(self):
        return self.file_stat.st_size if self.file_stat else 0

    @property
    def permission(self):
        return stat.S_IMODE(self.file_stat.st_mode) if self.file_stat else 0o777

    @property
    def owner(self):
        return self.file_stat.st_uid if self.file_stat else -1

    @property
    def group_owner(self):
        return self.file_stat.st_gid if self.file_stat else -1

    @property
    def is_owner(self):
        return self.owner == (os.geteuid() if hasattr(os, "geteuid") else 0)

    @property
    def is_group_owner(self):
        return self.group_owner in (os.getgroups() if hasattr(os, "getgroups") else [0])

    @property
    def is_user_or_group_owner(self):
        return self.is_owner or self.is_group_owner

    @property
    def is_user_and_group_owner(self):
        return self.is_owner and self.is_group_owner

    def is_basically_legal(self, perm='none', strict_permission=True):
        if sys.platform.startswith("win"):
            return self.check_windows_permission(perm)
        else:
            return self.check_linux_permission(perm, strict_permission=strict_permission)

    def check_linux_permission(self, perm='none', strict_permission=True):
        if not self.is_exists and perm != 'write':
            logger.error(f"path: {self.file} not exist, please check if file or dir is exist")
            return False
        if self.is_softlink:
            logger.error(f"path :{self.file} is a soft link, not supported, please import file(or directory) directly")
            solution_log(SOLUTION_BASE_LOC + SOFT_LINK_SUB_CHAPTER)
            return False
        if not self.is_user_or_group_owner and self.is_exists:
            logger.error(f"current user isn't path:{self.file}'s owner or ownergroup")
            solution_log(SOLUTION_BASE_LOC + OWNER_SUB_CHAPTER)
            return False
        if perm == 'read':
            if strict_permission and self.permission & READ_FILE_NOT_PERMITTED_STAT > 0:
                logger.error(f"The file {self.file} is group writable, or is others writable, "
                             "as import file(or directory) permission should not be over 0o755(rwxr-xr-x)")
                solution_log(SOLUTION_BASE_LOC + PERMISSION_SUB_CHAPTER)
                return False
            if not os.access(self.realpath, os.R_OK) or self.permission & stat.S_IRUSR == 0:
                logger.error(f"Current user doesn't have read permission to the file {self.file}, "
                             "as import file(or directory) permission should be at least 0o400(r--------) ")
                solution_log(SOLUTION_BASE_LOC + PERMISSION_SUB_CHAPTER)
                return False
        elif perm == 'write' and self.is_exists:
            if (strict_permission or self.is_file) and self.permission & WRITE_FILE_NOT_PERMITTED_STAT > 0:
                logger.error(f"The file {self.file} is group writable, or is others writable, "
                             "as export file(or directory) permission should not be over 0o755(rwxr-xr-x)")
                solution_log(SOLUTION_BASE_LOC + PERMISSION_SUB_CHAPTER)
                return False
            if not os.access(self.realpath, os.W_OK):
                logger.error(f"Current user doesn't have write permission to the file {self.file}, "
                             "as export file(or directory) permission should be at least 0o200(-w-------) ")
                solution_log(SOLUTION_BASE_LOC + PERMISSION_SUB_CHAPTER)
                return False
        return True

    def check_windows_permission(self, perm='none'):
        if not self.is_exists and perm != 'write':
            logger.error(f"path: {self.file} not exist, please check if file or dir is exist")
            return False
        if self.is_softlink:
            logger.error(f"path :{self.file} is a soft link, not supported, please import file(or directory) directly")
            solution_log(SOLUTION_BASE_LOC + SOFT_LINK_SUB_CHAPTER)
            return False
        return True

    def is_legal_file_size(self, max_size):
        if not self.is_file:
            logger.error(f"path: {self.file} is not a file")
            return False
        if self.file_size > max_size:
            logger.error(f"file_size:{self.file_size} byte out of max limit {max_size} byte")
            return False
        else:
            return True

    def is_legal_file_type(self, file_types: list):
        if not self.is_file and self.is_exists:
            logger.error(f"path: {self.file} is not a file")
            return False
        for file_type in file_types:
            if os.path.splitext(self.file)[1] == f".{file_type}":
                return True
        logger.error(f"path:{self.file}, file type not in {file_types}")
        return False


def ms_open(file, mode="r", max_size=None, softlink=False, write_permission=PERMISSION_NORMAL, **kwargs):
    file_stat = FileStat(file)

    if file_stat.is_exists and file_stat.is_dir:
        raise OpenException(f"Expecting a file, but it's a folder. {file}")

    if "r" in mode:
        if not file_stat.is_exists:
            raise OpenException(f"No such file or directory {file}")
        if max_size is None:
            raise OpenException(f"Reading files must have a size limit control. {file}")
        if max_size != MAX_SIZE_UNLIMITE and max_size < file_stat.file_size:
            raise OpenException(f"The file size has exceeded the specifications and cannot be read. {file}")

    if "w" in mode:
        if file_stat.is_exists and not file_stat.is_owner:
            raise OpenException(
                f"The file owner is inconsistent with the current process user and is not allowed to write. {file}"
            )
        if file_stat.is_exists:
            os.remove(file)

    if not softlink and file_stat.is_softlink:
        raise OpenException(f"Softlink is not allowed to be opened. {file}")

    if "a" in mode:
        if not file_stat.is_owner:
            raise OpenException(
                f"The file owner is inconsistent with the current process user and is not allowed to write. {file}"
            )
        if file_stat.permission != (file_stat.permission & write_permission):
            os.chmod(file, file_stat.permission & write_permission)

    if "+" in mode:
        flags = os.O_RDONLY | os.O_RDWR
    elif "w" in mode or "a" in mode or "x" in mode:
        flags = os.O_RDONLY | os.O_WRONLY
    else:
        flags = os.O_RDONLY

    if "w" in mode or "x" in mode:
        flags = flags | os.O_TRUNC | os.O_CREAT
    if "a" in mode:
        flags = flags | os.O_APPEND | os.O_CREAT
    return os.fdopen(os.open(file, flags, mode=write_permission), mode, **kwargs)


class UmaskWrapper:
    """Write with preset umask
    >>> with UmaskWrapper():
    >>>     ...
    """

    def __init__(self, umask=0o027):
        self.umask, self.ori_umask = umask, None

    def __enter__(self):
        self.ori_umask = os.umask(self.umask)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        os.umask(self.ori_umask)
