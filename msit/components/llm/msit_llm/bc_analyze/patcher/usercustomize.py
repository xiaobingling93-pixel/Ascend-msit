import atexit
import os

from ascend.model_test_patcher import patch_model_test, MODEL_TEST_INSTANCE
from patcher_log import logger as patcher_logger
from msit_llm.bc_analyze import Synthesizer


patcher_logger.debug("Patcher script entered.", stack_info=True)
patch_model_test()
patcher_logger.debug("Model Test Patched.")

def save_before_exit():
    if MODEL_TEST_INSTANCE is not None and hasattr(MODEL_TEST_INSTANCE, 'csv_debug'):
        csv_dict = getattr(MODEL_TEST_INSTANCE, 'csv_debug')
        
        try:
            synthezier = Synthesizer(
                queries=csv_dict.get('queries'),
                input_token_ids=csv_dict.get('input_token_ids'),
                output_token_ids=csv_dict.get('output_token_ids'),
                passed=csv_dict.get('pass')
            )

            temp_dir_name = os.environ['MSIT_TEMP_DIR_NAME']
            os.makedirs(temp_dir_name, 0o700, exist_ok=True)
            os.chdir(temp_dir_name)

            synthezier.to_csv()
        except Exception as e:
            patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
            
atexit.register(save_before_exit)
