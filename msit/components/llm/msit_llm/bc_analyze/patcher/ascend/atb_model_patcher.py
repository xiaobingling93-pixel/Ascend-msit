import re
from unittest.mock import patch

import torch

from msit_llm.common.log import logger as msit_logger
from msit_llm.bc_analyze.patcher.patcher_log import logger as patcher_logger

try:
    from atb_llm.runner.model_runner import ModelRunner
except ModuleNotFoundError:
    err_msg = "Either 'ATB Speed' environment is not properly set," \
              "or the 'ATB Speed' directories has been runined." \
              "Please try to reinstall 'ATB Speed' to solve that"
    msit_logger.error(err_msg)
    raise


ATB_MODEL = None
BATCH_SIZE = None
INPUT_TOKEN_IDS = list()
PADDED_QUERIES = list()
QUERIES = list()

def patch_atb_model():
    """"""
    original_init = ModelRunner.__init__

    def init_patched(self, *args, **kwargs):
        """"""
        global ATB_MODEL
        try:
            original_init(self, *args, **kwargs)
        except Exception as e:
            patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
            patcher_logger.debug("ModelRunner.__init__ self: '%s'", self)
            patcher_logger.debug("ModelRunner.__init__ args: '%s'", args)
            patcher_logger.debug("ModelRunner.__init__ kwargs: '%s'", kwargs)
            raise
        
        ATB_MODEL = self
        patcher_logger.debug("ModelRunner instance attributes: '%s'", vars(ATB_MODEL))

        original_forward = ATB_MODEL.forward

        def atb_model_forward_patched(*args, **kwargs):
            try:
                logits = original_forward(*args, **kwargs)
            except Exception as e:
                patcher_logger.debug("Exception info: '%s'", e, stack_info=True)
                patcher_logger.debug("Runner.forward args: '%s'", args)
                kwargs.pop('kv_cache')
                patcher_logger.debug("Runner.forward kwargs: '%s'", kwargs)
                raise

            input_ids = kwargs.get('input_ids').cpu()
            is_prefill = kwargs.get('is_prefill')
            input_lengths = kwargs.get('input_lengths').cpu()

            global BATCH_SIZE
            BATCH_SIZE = len(input_lengths)

            if is_prefill:
                accumulated_seq_len = torch.cumsum(input_lengths, dim=0)
                padded_seq_len = torch.cat((torch.tensor([0]), accumulated_seq_len), dim=0)

                for i in range(1, BATCH_SIZE + 1):
                    input_token_ids = input_ids[padded_seq_len[i-1] : padded_seq_len[i]]

                    from ..hgface import TOKENIZER_DECODE
                    if TOKENIZER_DECODE is None:
                        patcher_logger.debug('Tokenizer.decode is None: %s', TOKENIZER_DECODE)
                        raise RuntimeError
                    
                    padded_queries = TOKENIZER_DECODE(input_token_ids)
                    
                    all_queries = re.findall(r'\n\n(.*?Answer:)', padded_queries, re.DOTALL)
                    if all_queries:
                        global INPUT_TOKEN_IDS
                        INPUT_TOKEN_IDS.append(input_token_ids)
                        
                        global PADDED_QUERIES
                        PADDED_QUERIES.append(padded_queries)

                        global QUERIES
                        QUERIES.append(all_queries[-1])

            return logits

        patch.object(ATB_MODEL, 'forward', new=atb_model_forward_patched).start()
    
    patch.object(ModelRunner, '__init__', new=init_patched).start()
