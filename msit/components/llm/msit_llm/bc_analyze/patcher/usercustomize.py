
import atexit
import os

from msit_llm.common.log import logger as msit_logger
from patcher_log import logger as patcher_logger
patcher_logger.debug("Patcher script entered.")

try:
    from ascend.model_test_patcher import patch_model_test
except Exception as e:
    patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
    raise

patch_model_test()
patcher_logger.debug("Model Test Patched.")
# delimiter
patcher_logger.debug("-" * 50)


def save_before_exit():
    # should be imported after exit, because first import it will be 'None' and not gonna updated
    from ascend.model_test_patcher import MODEL_TEST_INSTANCE
    patcher_logger.debug("MODEL_TEST_INSTANCE: '%s'", MODEL_TEST_INSTANCE)

    # catch all statements, and store the debug info
    try:
        csv_dict = getattr(MODEL_TEST_INSTANCE, 'csv_debug')

        from msit_llm.bc_analyze import Synthesizer

        synthezier = Synthesizer(
            queries=csv_dict.get('queries'),
            input_token_ids=csv_dict.get('input_token_ids'),
            output_token_ids=csv_dict.get('output_token_ids'),
            passed=csv_dict.get('pass')
        )

        temp_dir_name = os.environ['MSIT_TEMP_DIR_NAME']
        patcher_logger.debug("MSIT_TEMP_DIR_NAME: '%s'", temp_dir_name)

        msit_logger.info("Creating temporary directory for saving running command results...")
        os.makedirs(temp_dir_name, 0o700, exist_ok=True)
        msit_logger.info("Creating temporary directory '%s': Success", temp_dir_name)

        os.chdir(temp_dir_name)
        msit_logger.info("The result collected by 'Synthesizer' from the running command will be saved under", os.getcwd())

        synthezier.to_csv()
    except Exception as e:
        # raise is not gonna work inside usercustomize especially at exit
        patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
    
            
atexit.register(save_before_exit)
