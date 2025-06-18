# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os

from pathlib import Path

from ascend_utils.common.security import get_valid_read_path
from msmodelslim.infra.practice_manager import NaiveQuantization
from msmodelslim.tools import logger as msmodelslim_logger
from msmodelslim.app.naive_quantization.quantization import Quantization as quant_backend


class NaiveEntrance:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        practice_lab_dir = os.path.abspath(os.path.join(cur_dir, '../../practice_lab'))
        practice_lab_dir = get_valid_read_path(practice_lab_dir, is_dir=True)
        self.config_dir = Path(practice_lab_dir)

        self.naive_quantizer = NaiveQuantization(self.config_dir)

    def run_quantization(self, args):
        try:
            # Get best configuration
            best_config = self.naive_quantizer.get_best_practice(
                model_type=args.model_type,
                config_path=args.config_path,
                quant_type=args.quant_type,
                device=args.device,
                model_path=args.model_path,
                save_path=args.save_path,
                trust_remote_code=args.trust_remote_code
            )

            quant_example = quant_backend()
            quant_example.quant_process(best_config)
            return best_config

        except ValueError as e:
            msmodelslim_logger.logger_error(f"Error: {e}")
            return None

