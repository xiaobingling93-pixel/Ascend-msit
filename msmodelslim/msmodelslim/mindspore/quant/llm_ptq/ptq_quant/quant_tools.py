# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import time

import mindspore as ms
from mindspore.communication import get_rank
from mindspore.dataset.engine.datasets import RepeatDataset

from ascend_utils.common.security import check_type
from msmodelslim import logger
from msmodelslim.mindspore.quant.llm_ptq.ptq_quant import QuantConfig
from msmodelslim.mindspore.quant.llm_ptq.mindspore_gs.ptq import PTQMode, PTQConfig, OutliersSuppressionType
from msmodelslim.mindspore.quant.llm_ptq.mindspore_gs.common import BackendTarget
from msmodelslim.mindspore.quant.llm_ptq.mindspore_gs.ptq.ptq import PTQ


class Calibrator(object):
    def __init__(self,
                 model,
                 cfg:QuantConfig,
                 calib_data=None,
                 ) -> None:

        check_type(cfg, QuantConfig, param_name="cfg")
        if calib_data is not None:
            check_type(calib_data, RepeatDataset, param_name="calib_data")
        self.cfg = cfg
        self.logger = logger
        self.calib_data = calib_data
        outliers_suppression = OutliersSuppressionType.SMOOTH \
            if self.cfg.do_smooth else OutliersSuppressionType.NONE

        cfg = PTQConfig(
            mode=PTQMode.QUANTIZE,
            backend=BackendTarget.ASCEND,
            weight_quant_dtype=self.cfg.w_bit,
            act_quant_dtype=self.cfg.a_bit,
            kvcache_quant_dtype=self.cfg.use_kvcache_quant,
            outliers_suppression=outliers_suppression,
            opname_blacklist=self.cfg.disable_names
        )
       
        self.ptq = PTQ(config=cfg)
        self.network = model
        self.helper = self.cfg.msconfig

    def run(self):
        start = time.time()
        self.network = self.quant_net(self.network, self.helper, self.ptq, self.calib_data)

    def quant_net(self, net, network_helper, ptq, ds):
        """Quant network with algorithm."""
        quant_start = time.time()
        self.logger.info('Quantize-ing network...')
        start_time = time.time()
        ptq.apply(net, network_helper, ds)
        self.logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
        start_time = time.time()
        net.phase = "quant_convert"
        ptq.convert(net)
        self.logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
        self.logger.info(f'Quant Network cost total time is {time.time() - quant_start} s.')
        return net

    def save(self, save_path):
        check_type(save_path, str, param_name="save_path")
        start = time.time()
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        save_ckpt_path = os.path.join(save_path, f"rank_{rank_id}")
        os.makedirs(name=save_ckpt_path, exist_ok=True)
        ms.save_checkpoint(self.network.parameters_dict(), os.path.join(save_ckpt_path, f"ptq.ckpt"),
                           choice_func=lambda
                               x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
        self.logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
        self.logger.info(f'Checkpoint saved to {save_ckpt_path}')
