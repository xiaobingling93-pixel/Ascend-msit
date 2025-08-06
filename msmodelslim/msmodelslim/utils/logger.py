# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
import logging
from functools import wraps
from logging import Logger

from typing_extensions import Union

from msmodelslim.utils.exception import SchemaValidateError


class MsgConst:
    """
    Class for log messages const
    """
    SPECIAL_CHAR = ["\n", "\r", "\u007F", "\b", "\f", "\t", "\u000B", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]


def filter_special_chars(func):
    @wraps(func)
    def func_level(msg, *args):
        for char in MsgConst.SPECIAL_CHAR:
            if isinstance(msg, str):
                msg = msg.replace(char, ' ')
        return func(msg, *args)

    return func_level


def filter_logger(cur_logger: Logger):
    setattr(cur_logger, 'critical', filter_special_chars(cur_logger.critical))
    setattr(cur_logger, 'debug', filter_special_chars(cur_logger.debug))
    setattr(cur_logger, 'error', filter_special_chars(cur_logger.error))
    setattr(cur_logger, 'info', filter_special_chars(cur_logger.info))
    setattr(cur_logger, 'warning', filter_special_chars(cur_logger.warning))


def get_logger(name: str = ''):
    if not name:
        return get_root_logger()
    cur_logger = logging.getLogger(name)
    filter_logger(cur_logger)
    return cur_logger


def get_root_logger():
    root_logger = logging.getLogger(__name__.split('.')[0])
    root_logger.propagate = False
    root_logger.setLevel(logging.INFO)
    filter_logger(root_logger)
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    return root_logger


logger = get_root_logger()

LOG_LEVEL = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL
}

LOGGER_FUNC = {
    "debug": lambda msg: logger.debug(msg),
    "info": lambda msg: logger.info(msg),
    "warn": lambda msg: logger.warning(msg),
    "warning": lambda msg: logger.warning(msg),
    "error": lambda msg: logger.error(msg),
    "critical": lambda msg: logger.critical(msg),
}

LOGGER_LEVEL_FUNC_MAP = {
    "debug": 'debug',
    "info": 'info',
    "warn": 'warning',
    "warning": 'warning',
    "error": 'error',
    "critical": 'critical',
}


def get_logger_func(level: str, cur_logger: Logger = logger):
    return getattr(cur_logger, LOGGER_LEVEL_FUNC_MAP.get(level.lower()))


def set_logger_level(level="info"):
    if not isinstance(level, str):
        raise SchemaValidateError(f"level must be str, not {type(level)}",
                                  action='Please make sure log level is a string')
    if level.lower() in LOG_LEVEL:
        logger.setLevel(LOG_LEVEL.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


def progress_bar(iterable, desc: str = None, total: int = -1, interval: int = 1):
    if total == -1 and hasattr(iterable, "__len__"):
        total = len(iterable)

    format_str = "" if desc is None else (desc + ": ")
    if isinstance(total, int) and total > 0:
        format_str += "[%d/{}]".format(total)
    else:
        format_str += "[%d]"

    if not (isinstance(interval, int) and interval > 0):
        interval = 1

    prev_terminator = logging.StreamHandler.terminator
    logging.StreamHandler.terminator = '\r'
    for item_id, item in enumerate(iterable, start=1):
        if item_id % interval == 0:
            logger.info(format_str, item_id)
        yield item
    logging.StreamHandler.terminator = prev_terminator
    logger.info("")


def logger_setter(cur_logger: Union[str, Logger] = logger):
    if isinstance(cur_logger, str):
        cur_logger = get_logger(cur_logger)

    def decorator(cls):
        cls.logger = cur_logger
        return cls

    return decorator
