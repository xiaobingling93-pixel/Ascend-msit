from unittest.mock import patch

from transformers import AutoTokenizer

from msit_llm.bc_analyze.patcher.patcher_log import logger as patcher_logger


TOKENIZER = None
TOKENIZER_DECODE = None
OUTPUT_TOKEN_IDS = list()
MODEL_OUTPUT = list()

def patch_hgface_tokenizer():
    """AutoTokenizer.from_pretrained ---> from_pretrained_patched"""

    original_from_pretrained = AutoTokenizer.from_pretrained

    def from_pretrained_patched(*args, **kwargs):
        """"""
        global TOKENIZER
        try:
            TOKENIZER = original_from_pretrained(*args, **kwargs)
        except Exception as e:
            patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
            patcher_logger.debug("AutoTokenizer.from_pretrained args: '%s'", args)
            patcher_logger.debug("AutoTokenizer.from_pretrained kwargs: '%s'", kwargs)
            raise

        patcher_logger.debug("AutoTokenizer.from_pretrained returns: '%s'", TOKENIZER)
        
        global TOKENIZER_DECODE
        TOKENIZER_DECODE = TOKENIZER.decode
        def tokenizer_decode_patched(token_ids, *args, **kwargs):

            try:
                decoded_text = TOKENIZER_DECODE(token_ids, *args, **kwargs)
            except Exception as e:
                patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
                patcher_logger.debug("output_token_ids: '%s'", token_ids)
                patcher_logger.debug("tokenizer.decode args: '%s'", args)
                patcher_logger.debug("tokenizer.decode kwargs: '%s'", kwargs)
                raise

            patcher_logger.debug("tokenizer.decode returns: '%s'", decoded_text)

            # need to find another way
            from ..ascend import BATCH_SIZE
            if BATCH_SIZE > 0:

                global OUTPUT_TOKEN_IDS
                OUTPUT_TOKEN_IDS.append(token_ids)

                global MODEL_OUTPUT
                MODEL_OUTPUT.append(decoded_text)
                BATCH_SIZE -= 1
            
            return decoded_text

        patch.object(TOKENIZER, 'decode', new=tokenizer_decode_patched).start()

        return TOKENIZER

    patch('transformers.AutoTokenizer.from_pretrained', new=from_pretrained_patched).start()
