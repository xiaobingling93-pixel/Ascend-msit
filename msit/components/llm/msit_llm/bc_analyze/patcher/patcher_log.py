import logging
import os
from msit_llm.bc_analyze import get_timestamp


logger = logging.getLogger("msit_llm_patcher")
logger.setLevel(logging.DEBUG)
logger.propagate = False

logger_dir = os.path.expanduser('~/.msit_cache')
os.makedirs(logger_dir, 0o700, exist_ok=True)
file_name = os.path.join(logger_dir, 'msit_patcher.log')

file_handler = logging.FileHandler(
    filename=file_name,
    mode='w',
    encoding='utf-8',
)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s] [%(name)s]: '%(message)s'")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

os.chmod(file_name, 0o600)
