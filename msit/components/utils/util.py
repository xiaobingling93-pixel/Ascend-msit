# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import re

from components.utils.log import logger
from components.utils.constants import MAX_DATA_SIZE

STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")


def get_entry_points(entry_points_name):
    try:
        from importlib import metadata

        return metadata.entry_points().get(entry_points_name, [])
    except Exception:
        import pkg_resources

        return list(pkg_resources.iter_entry_points(entry_points_name))


def confirmation_interaction(prompt):
    confirm_pattern = re.compile(r'y(?:es)?', re.IGNORECASE)
    
    try:
        user_action = input(prompt)
    except Exception:
        return False
    
    return bool(confirm_pattern.match(user_action))


def load_file_to_read_common_check(value: str, max_size=MAX_DATA_SIZE, exts=None):
    if isinstance(exts, (tuple, list)):
        if not all(isinstance(ext, str) for ext in exts):
            logger.error("Expected type 'List[str]', got %r instead", exts)
            raise TypeError

        value_ext = os.path.splitext(value)[1]
        if all(value_ext != ext for ext in exts):
            logger.error("Expected extenstion to be one of %r, got %r instead", exts, value_ext)
            raise ValueError
        
    elif exts is not None:
        logger.error("Expected 'exts' to be 'List[str]', got %r instead", type(exts))
        raise TypeError
    
    if re.search(STR_WHITE_LIST_REGEX, value):
        logger.error("Invalid character: %r", value)
        raise ValueError
    
    # expand soft link
    value = os.path.realpath(value)
    
    # file name too long, file not exists, directory readable
    # no need to catch, argparse will handle that
    try:
        file_status = os.stat(value)
    except OSError as e:
        logger.error("%s: %r", e.strerror, value)
        raise
    
    # not regular file
    if not os.st.S_ISREG(file_status.st_mode):
        logger.error("Not a regular file: %r", value)
        raise ValueError
    
    confirmation_prompt = "The file %r is larger than expected. " \
                          "Attempting to read such a file could potentially impact system performance.\n" \
                          "Please confirm your awareness of the risks associated with this action ([y]/n): " % value
    
    if file_status.st_size > max_size and not confirmation_interaction(confirmation_prompt):
        logger.error("File too large: %r", value)
        raise ValueError
    
    # other writeable
    if (os.st.S_IWOTH & file_status.st_mode) == os.st.S_IWOTH:
        logger.error("Vulnerable path: %r should not be other writeable", value)
        raise PermissionError

    # uid
    cur_euid = os.geteuid()
    if file_status.st_uid != cur_euid:
        # not root
        if cur_euid != 0:
            logger.error("Inconsistent owner: %r", value)
            raise PermissionError
        
        # root but reading a other writeable file
        elif (os.st.S_IWGRP & file_status.st_mode) == os.st.S_IWGRP or \
             (os.st.S_IWUSR & file_status.st_mode) == os.st.S_IWUSR:
            logger.waring("Privilege escalation risk detected. Trying to read a file that belongs to"
                          " a normal user and is writeable to the user itself or the user group")

    return value