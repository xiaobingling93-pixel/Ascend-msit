from unittest.mock import patch

from transformers.models.auto.auto_factory import _BaseAutoModelClass


def patch_hgface_model():
    """"""
    original_from_pretrained = _BaseAutoModelClass.from_pretrained

    def from_pretrained_patched(*args, **kwargs):
        """"""
        model = original_from_pretrained(*args, **kwargs)
        return model
    
    patch.object('_BaseAutoModelClass', '__init__', new=from_pretrained_patched).start()