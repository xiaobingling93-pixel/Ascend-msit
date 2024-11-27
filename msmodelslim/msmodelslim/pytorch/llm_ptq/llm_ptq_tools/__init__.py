# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

__all__ = ['Calibrator', 'QuantConfig', 'FakeQuantizeCalibrator']

from .quant_config.quant_config import QuantConfig
from .quant_tools import Calibrator
from .calibrator.calibrator_classes.fakequantize_calibrator import FakeQuantizeCalibrator

