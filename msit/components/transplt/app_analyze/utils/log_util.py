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

import sys
import os
import stat
import logging
from logging import handlers

IS_PYTHON3 = sys.version_info > (3,)
LOG_FILE_PATH = "msit_transplt.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[%(lineno)d] - %(message)s"
LOG_LEVEL = {
    "notest": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL
}

OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR


def get_logger():
    inner_logger = logging.getLogger("msit transplt")
    inner_logger.propagate = False
    inner_logger.setLevel(logging.INFO)
    if not inner_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        stream_handler.setFormatter(formatter)
        inner_logger.addHandler(stream_handler)
    return inner_logger


def set_logger_level(level="info"):
    if level.lower() in LOG_LEVEL:
        logger.setLevel(LOG_LEVEL.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


def init_file_logger():
    for ii in logger.handlers:
        # Check if already set
        if isinstance(ii, handlers.TimedRotatingFileHandler) and os.path.basename(ii.stream.name) == LOG_FILE_PATH:
            return

    log_file_path = os.path.realpath(LOG_FILE_PATH)
    if os.path.exists(LOG_FILE_PATH):
        if not os.path.isfile(log_file_path):
            raise OSError(f"log file {log_file_path} exists, and may be a link or directory")
        os.remove(log_file_path)
    
    with os.fdopen(os.open(log_file_path, OPEN_FLAGS, OPEN_MODES), 'w') as log_file:
        pass

    # create console handler and formatter for logger
    fh = handlers.TimedRotatingFileHandler(log_file_path, when='midnight', interval=1, backupCount=7)
    formatter = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger = get_logger()