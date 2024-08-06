import os
import sys
from unittest.mock import patch

from msit_llm.common.log import logger as msit_logger
from msit_llm.bc_analyze.patcher.patcher_log import logger as patcher_logger


atb_speed_home_path = os.environ.get('ATB_SPEED_HOME_PATH')
if atb_speed_home_path is None:
    msit_logger.error("The environment which 'ATB Speed' requires is not properly set, try to source the related script first")
    raise RuntimeError

model_test_base_path = os.path.join(atb_speed_home_path, 'tests', 'modeltest')
if not os.path.exists(model_test_base_path):
    msit_logger.error("'modeltest' cannot be found, please reinstall 'ATB Speed'")
    raise FileNotFoundError

sys.path.append(model_test_base_path)
try:
    from base.model_test import ModelTest
except ModuleNotFoundError:
    msit_logger.error("There seems no 'ModelTest' class in 'model_test.py', please try to reinstall 'ATB Speed'")
    raise


MODEL_TEST_INSTANCE = None

def patch_model_test():
    """"""
    original_init = ModelTest.__init__

    def init_patched(self, *args, **kwargs):
        """"""
        try:
            original_init(self, *args, **kwargs)
        except Exception as e:
            patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
            patcher_logger.debug("ModelTest.__init__ args: '%s'", args)
            patcher_logger.debug("ModelTest.__init__ kwargs: '%s'", kwargs)
            raise
    
        global MODEL_TEST_INSTANCE
        MODEL_TEST_INSTANCE = self
        patcher_logger.debug("ModelTest instance attributes: '%s'", vars(MODEL_TEST_INSTANCE))
    
    patch.object(ModelTest, '__init__', new=init_patched).start()