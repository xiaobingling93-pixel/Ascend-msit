# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import atexit
import os

from msit_llm.common.log import logger
logger.info("Patcher script entered.")

try:
    from ascend.model_test_patcher import patch_model_test
except ModuleNotFoundError:
    logger.error(
        "'Model Test' patcher is not found. Please make sure 'model_test_patcher.py' exists"
    )
    exit(1) # raise in usercustomize will not stop the child process

patch_model_test()
logger.info("'Model Test' has successfully been patched.")


def save_before_exit():
    # should be imported after exit, because first import it will be 'None' and not gonna updated
    from ascend.model_test_patcher import MODEL_TEST_INSTANCE

    if MODEL_TEST_INSTANCE is None:
        return

    elif not hasattr(MODEL_TEST_INSTANCE, 'csv_debug'):
        logger.warning(
            "No 'csv_debug' is found, may result from latest updates in 'Model Test'"
        )
        return

    csv_dict = getattr(MODEL_TEST_INSTANCE, 'csv_debug')
    if not csv_dict:
        logger.error(
            "It seems that internally errors occured inside the 'Model Test', please "
            "try to run the command without 'msit' to see if it works independently"
        )
    
    from msit_llm.bc_analyze import Synthesizer

    synthezier = Synthesizer(
        queries=csv_dict.get('queries'),
        input_token_ids=csv_dict.get('input_token_ids'),
        output_token_ids=csv_dict.get('output_token_ids'),
        passed=csv_dict.get('pass')
    )

    temp_dir_name = os.environ['MSIT_TEMP_DIR_NAME']

    logger.info("Creating temporary directory for saving running command results...")
    os.makedirs(temp_dir_name, 0o700, exist_ok=True)
    logger.info("Creating temporary directory '%s': Success", temp_dir_name)

    os.chdir(temp_dir_name)
    logger.info(
        "The result collected by 'Synthesizer' from the running command will be saved under '%s'", 
        os.getcwd()
    )

    synthezier.to_csv()

    return


atexit.register(save_before_exit)
