from unittest.mock import patch

from msit_llm.common.log import logger

try:
    from atb_llm.runner.model_runner import ModelRunner
except ModuleNotFoundError:
    logger.error("qwewqewqeq")
    raise


def patch_atb_model():
    """"""
    original_init = ModelRunner.__init__

    def init_patched(self, *args, **kwargs):
        """"""
        pass

    patch.object('ModelRunner', '__init__', new=init_patched).start()
    