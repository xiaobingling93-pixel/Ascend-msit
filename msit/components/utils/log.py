# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import logging


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL
}

SPECIAL_CHAR = ["\n", "\r", "\u007F", "\b", "\f", "\t", "\u000B", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]

LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


def set_log_level(level="info"):
    if level.lower() in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


def set_logger(msit_logger):
    msit_logger.propagate = False
    msit_logger.setLevel(logging.INFO)
    if not msit_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        msit_logger.addHandler(stream_handler)


def get_filter_handle(handle, self):
    def filter_handle(self, record):
        for char in SPECIAL_CHAR:
            record.msg = record.msg.replace(char, '_')
        return handle(record)

    return filter_handle.__get__(self, type(self))


logger = logging.getLogger("msit_logger")
set_logger(logger)
if hasattr(logger, 'handle'):
    logger.handle = get_filter_handle(logger.handle, logger)
else:
    raise RuntimeError('The Python version is not suitable')


def msg_filter(msg):
    if not isinstance(msg, str):
        raise RuntimeError('msg type is not string, please check.')
    for char in SPECIAL_CHAR:
        msg = msg.replace(char, '_')
    return msg
