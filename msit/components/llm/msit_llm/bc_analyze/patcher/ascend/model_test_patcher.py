import os
import sys
from unittest.mock import patch

from msit_llm.common.log import logger


atb_speed_home_path = os.environ.get('ATB_SPEED_HOME_PATH')
if atb_speed_home_path is None:
    logger.error("The environment which 'ATB Speed' requires is not properly set, try to source the related script first")
    raise RuntimeError

model_test_base_path = os.path.join(atb_speed_home_path, 'tests', 'modeltest')
if not os.path.exists(model_test_base_path):
    logger.error("'modeltest' cannot be found, please reinstall 'ATB Speed'")
    raise FileNotFoundError

sys.path.append(model_test_base_path)
try:
    from base_model_test import ModelTest
except ModuleNotFoundError:
    logger.error()
    raise


MODEL_TEST_INSTANCE = None

def patch_model_test():
    """"""
    original_init = ModelTest.__init__

    def init_patched(self, *args, **kwargs):
        """"""
        global MODEL_TEST_INSTANCE
        MODEL_TEST_INSTANCE = self
        original_init(self, *args, **kwargs)
    
    patch.object(ModelTest, '__init__', new=init_patched).start()