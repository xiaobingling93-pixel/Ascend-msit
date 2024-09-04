import logging


def get_logger():
    llm_logger = logging.getLogger("msit_debug_logger")
    llm_logger.propagate = False
    llm_logger.setLevel(logging.INFO)
    if not llm_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        llm_logger.addHandler(stream_handler)
    return llm_logger


logger = get_logger()


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL
}