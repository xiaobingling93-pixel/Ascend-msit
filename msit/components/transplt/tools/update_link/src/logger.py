import logging

LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "error": logging.ERROR,
}


def set_logger_level(level="info"):
    if level.lower() in LOG_LEVEL:
        logger.setLevel(LOG_LEVEL.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


def get_logger():
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s : %(levelname)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    inner_logger = logging.getLogger(__name__)
    return inner_logger


logger = get_logger()
