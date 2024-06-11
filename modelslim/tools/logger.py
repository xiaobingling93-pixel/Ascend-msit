# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
import logging


def get_logger():
    amc_logger = logging.getLogger("modelslim-logger")
    amc_logger.propagate = False
    amc_logger.setLevel(logging.INFO)
    if not amc_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        amc_logger.addHandler(stream_handler)
    return amc_logger


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


def set_logger_level(level="info"):
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


logger = get_logger()