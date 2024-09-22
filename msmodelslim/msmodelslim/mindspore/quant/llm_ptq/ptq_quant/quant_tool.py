# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

"""Quant network."""
import os
import argparse
import time

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore.communication import get_rank

from mindformers import MindFormerConfig

from mindspore_gs.ptq import PTQMode, PTQConfig, OutliersSuppressionType
from mindspore_gs.common import BackendTarget, logger

from mindspore_gs.ptq.ptq import PTQ

from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper


MAX_READ_FILE_SIZE_20G = 20 * 1024 * 1024 * 1024


class Calibrator(object):
    def __init__(self,
                 cfg,
                 config_path="",
                 calib_data=None,
                 ) -> None:
        # validation check
        #check_type(cfg, QuantConfig, param_name="cfg")

        # initialization
        self.cfg = cfg
        self.calib_data = calib_data
        self.rollback_names = self.cfg.disable_names
        self.config_path = config_path
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, 
                        backend=BackendTarget.ASCEND, 
                        weight_quant_dtype=self.dtype_formatter(self.cfg.w_bit),
                        act_quant_dtype=self.dtype_formatter(self.cfg.a_bit), 
                        kvcache_quant_dtype=self.dtype_formatter(self.cfg.use_kvcache_quant),
                        outliers_suppression=OutliersSuppressionType.SMOOTH, 
                        opname_blacklist=['lm_head', 'w2'])
        self.ptq = PTQ(config=cfg)
        self.helper = MFParallelLlama2Helper(self.config_path)
        self.network = self.helper.create_network()
    
    def dtype_formatter(self, name: str):
            if name == 'int8':
                return msdtype.int8
            return None

    def run(self):
        start = time.time()
        logger.info(f'Create Network cost time is {time.time() - start} s.')
        self.network = self.quant_net(self.network, self.helper, self.ptq, self.calib_data)

    def quant_net(self, net, network_helper, ptq, ds):
        """Quant network with algorithm."""
        quant_start = time.time()
        logger.info('Quantize-ing network...')
        start_time = time.time()
        ptq.apply(net, network_helper, ds)
        logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
        start_time = time.time()
        net.phase = "quant_convert"
        ptq.convert(net)
        logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
        logger.info(f'Quant Network cost total time is {time.time() - quant_start} s.')
        return net

    def save(self, save_ckpt_path=""):
        start = time.time()
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        ms.save_checkpoint(self.network .parameters_dict(), os.path.join(save_path, f"ptq.ckpt"),
                           choice_func=lambda
                               x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
        logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
        logger.info(f'Checkpoint saved to {save_path}...')
