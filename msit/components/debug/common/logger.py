import logging


def get_logger():
    debug_logger = logging.getLogger("msit_debug_logger")
    debug_logger.propagate = False
    debug_logger.setLevel(logging.INFO)
    if not debug_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        debug_logger.addHandler(stream_handler)
    return debug_logger


logger = get_logger()


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL
}