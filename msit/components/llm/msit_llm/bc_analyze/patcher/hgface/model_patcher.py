from unittest.mock import patch

from transformers.models.auto.auto_factory import _BaseAutoModelClass

from msit_llm.bc_analyze.patcher.patcher_log import logger as patcher_logger


MODEL = None

def patch_hgface_model():
    """"""
    original_from_pretrained = _BaseAutoModelClass.from_pretrained

    def from_pretrained_patched(*args, **kwargs):
        """"""
        global MODEL
        try:  
            MODEL = original_from_pretrained(*args, **kwargs)
        except Exception as e:
            patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
            patcher_logger.debug("_BaseAutoModelClass.from_pretrained args: '%s'", args)
            patcher_logger.debug("_BaseAutoModelClass.from_pretrained kwargs: '%s'", kwargs)
            raise
        
        return MODEL
    
    patch.object('_BaseAutoModelClass', '__init__', new=from_pretrained_patched).start()