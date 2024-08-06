import atexit
import os

from ascend.model_test_patcher import patch_model_test, MODEL_TEST_INSTANCE
from msit_llm.bc_analyze import Synthesizer

patch_model_test()

def save_before_exit():
    if MODEL_TEST_INSTANCE is None and hasattr(MODEL_TEST_INSTANCE, 'csv_debug'):
        csv_dict = getattr(MODEL_TEST_INSTANCE, 'csv_debug')

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

atexit.register(save_before_exit)
