import logging
import os


logger = logging.getLogger("msit_llm_patcher")
logger.setLevel(logging.DEBUG)

logger_dir = os.path.expanduser('~/.msit_cache')
os.makedirs(logger_dir, 0o500, exist_ok=True)
file_name = os.path.join(logger_dir, 'patcher.log')

file_handler = logging.FileHandler(
    filename=file_name,
    mode='w',
    encoding='utf-8',
)
formatter = logging.Formatter('[%(asctime)s]: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
