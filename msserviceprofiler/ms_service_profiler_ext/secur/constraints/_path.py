# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import re
import os
import stat
import threading

from .base import BaseConstraint
from ..utils import get_file_size_config


FILE_STAT_CACHES = {}
FILE_STAT_LOCK = threading.Lock()


def fetch_from_cache(path: str):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"'path' must be str or os.PathLike. Got {type(path).__name__} instead")

    with FILE_STAT_LOCK:
        if path in FILE_STAT_CACHES:
            return FILE_STAT_CACHES[path]

        try:
            stat_result = os.stat(path, follow_symlinks=False)
            abs_path = os.path.abspath(path)
            FILE_STAT_CACHES[path] = FILE_STAT_CACHES[abs_path] = stat_result
        except OSError:
            FILE_STAT_CACHES[path] = None

        return FILE_STAT_CACHES[path]


class Exists(BaseConstraint):
    description = "an existing path"

    def is_satisfied_by(self, path):
        return fetch_from_cache(path) is not None


class IsFile(BaseConstraint):
    description = "a regular file"

    def is_satisfied_by(self, path):
        stat_result = fetch_from_cache(path)
        return stat_result is not None and stat.S_ISREG(stat_result.st_mode)


class IsDir(BaseConstraint):
    description = "a directory"

    def is_satisfied_by(self, path):
        stat_result = fetch_from_cache(path)
        return stat_result is not None and stat.S_ISDIR(stat_result.st_mode)


class HasSoftLink(BaseConstraint):
    """Check if path contains a soft link; if path does not exist, raise FileNotFoundError."""
    description = "a soft link"

    def is_satisfied_by(self, path):
        norm_path = os.path.normpath(os.path.abspath(path))
        components = norm_path.split(os.sep)

        current = os.path.sep
        for part in components[1:]:
            if not part:
                continue
            current = os.path.join(current, part)
            current_stat = fetch_from_cache(current)
            if current_stat is None:
                raise FileNotFoundError(f"Path component {current!r} does not exist.")
            if stat.S_ISLNK(current_stat.st_mode):
                return True

        final_stat = fetch_from_cache(norm_path)
        if final_stat is None:
            raise FileNotFoundError(f"Path {norm_path!r} does not exist.")

        return stat.S_ISLNK(final_stat.st_mode)


class IsReadable(BaseConstraint):
    description = "a readable path"

    def is_satisfied_by(self, path):
        return os.access(path, os.R_OK)


class IsWritable(BaseConstraint):
    description = "a writable path"

    def is_satisfied_by(self, path):
        return os.access(path, os.W_OK)


class IsExecutable(BaseConstraint):
    description = "an executable path"

    def is_satisfied_by(self, path):
        return os.access(path, os.X_OK)


class IsWritableToGroupOrOthers(BaseConstraint):
    description = "writable to group or others"

    def is_satisfied_by(self, path):
        stat_result = fetch_from_cache(path)
        if stat_result is None:
            raise FileNotFoundError(f"The path {path!r} does not exist or cannot be accessed.")

        return bool(stat_result.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


class IsConsistentToCurrentUser(BaseConstraint):
    description = "consistent with the current user"

    def is_satisfied_by(self, path):
        stat_result = fetch_from_cache(path)
        if stat_result is None:
            raise FileNotFoundError(f"The path {path!r} does not exist or cannot be accessed.")

        return stat_result.st_uid == os.geteuid()


class IsSizeReasonable(BaseConstraint):
    description = "reasonable on its size"

    def is_satisfied_by(self, path):
        stat_result = fetch_from_cache(path)
        if stat_result is None:
            raise FileNotFoundError(f"The path {path!r} does not exist or cannot be accessed.")

        if not stat.S_ISREG(stat_result.st_mode):
            raise OSError(f"The path {path!r} is not a regular file.")

        return self._check_size(path, stat_result.st_size)

    def _check_size(self, path, file_size):
        file_size_config = get_file_size_config()
        ext_size_mapping = file_size_config['ext_mapping']
        require_confirm = file_size_config['require_confirm']

        ext = os.path.splitext(path)[1]
        size_limit = ext_size_mapping.get(ext, max(ext_size_mapping.values()))
        is_reasonable = file_size < size_limit

        if not is_reasonable and require_confirm:
            return self._prompt_user_confirmation(path, file_size)

        return is_reasonable

    def _prompt_user_confirmation(self, path, file_size):
        prompt = (
            f"The file {path!r} is larger than expected ({file_size} Bytes). This could be a potential security threat."
            " Attempting to read such a file could potentially impact system performance.\n"
            "Please confirm your awareness of the risks associated with this action (y/n): "
        )
        confirm_pattern = re.compile(r'y(?:es)?', re.IGNORECASE)

        try:
            user_action = input(prompt)
        except Exception:
            return False

        return bool(confirm_pattern.match(user_action))
