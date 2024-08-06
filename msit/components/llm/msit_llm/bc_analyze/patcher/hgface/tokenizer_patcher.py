import os
import re
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from ....common.log import logger


def patch_tokenizer():
    """AutoTokenizer.from_pretrained ---> from_pretrained_patched"""


    original_from_pretrained = AutoTokenizer.from_pretrained

    def from_pretrained_patched(*args, **kwargs):
        """"""
        pass

    patch('transformers.AutoTokenizer.from_pretrained', new=from_pretrained_patched).start()
