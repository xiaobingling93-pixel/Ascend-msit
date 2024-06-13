# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd. All rights reserved.
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

import logging


def get_logger(name="auto-optimizer-logger"):
    auto_logger = logging.getLogger(name)
    auto_logger.propagate = False
    auto_logger.setLevel(logging.INFO)
    if not auto_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        auto_logger.addHandler(stream_handler)
    return auto_logger


LOG_LEVEL = {
    "notest": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}


def set_logger_level(level="info"):
    if level.lower() in LOG_LEVEL:
        logger.setLevel(LOG_LEVEL.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


logger = get_logger()
