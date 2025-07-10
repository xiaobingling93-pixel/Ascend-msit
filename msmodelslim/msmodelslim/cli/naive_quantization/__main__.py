# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import functools
import os
from pathlib import Path

from ascend_utils.common.security.path import get_valid_read_path
from msmodelslim.app.base import BaseModel
from msmodelslim.model import ModelFactory
from msmodelslim.app.quant_service import ModelslimV0QuantService
from msmodelslim.app.naive_quantization import NaiveQuantizationApplication
from msmodelslim.infra.dataset_loader import FileDatasetLoader
from msmodelslim.infra.practice_manager import PracticeManager


def get_practice_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    practice_lab_dir = os.path.abspath(os.path.join(cur_dir, '../../practice_lab'))
    practice_lab_dir = get_valid_read_path(practice_lab_dir, is_dir=True)
    return Path(practice_lab_dir)


def get_dataset_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calib_lab_dir = os.path.abspath(os.path.join(cur_dir, '../../calib_lab'))
    calib_lab_dir = get_valid_read_path(calib_lab_dir, is_dir=True)
    return Path(calib_lab_dir)


def main(args):
    config_dir = get_practice_dir()
    practice_manager = PracticeManager(official_config_dir=config_dir)
    dataset_dir = get_dataset_dir()
    dataset_loader = FileDatasetLoader(dataset_dir)
    quant_service = ModelslimV0QuantService(dataset_loader)
    model_factory = functools.partial(ModelFactory.create, interface=BaseModel)

    app = NaiveQuantizationApplication(
        practice_manager=practice_manager,
        quant_service=quant_service,
        model_factory=model_factory)
    app.quant(model_type=args.model_type, model_path=args.model_path, save_path=args.save_path, device=args.device,
            quant_type=args.quant_type)
