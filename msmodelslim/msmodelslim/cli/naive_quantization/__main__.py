# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
from pathlib import Path

from msmodelslim.app.naive_quantization import NaiveQuantizationApplication
from msmodelslim.app.quant_service.proxy import QuantServiceProxy
from msmodelslim.core.base.model import BaseModelInterface
from msmodelslim.infra.dataset_loader import FileDatasetLoader
from msmodelslim.infra.practice_manager import PracticeManager
from msmodelslim.model import ModelFactory
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.utils.security.path import get_valid_read_path


def get_practice_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_practice_dir = os.path.abspath(os.path.join(cur_dir, '../../lab_practice'))
    lab_practice_dir = get_valid_read_path(lab_practice_dir, is_dir=True)
    return Path(lab_practice_dir)


def get_dataset_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_calib_dir = os.path.abspath(os.path.join(cur_dir, '../../lab_calib'))
    lab_calib_dir = get_valid_read_path(lab_calib_dir, is_dir=True)
    return Path(lab_calib_dir)


def create_model(model_type: str, model_path: Path, trust_remote_code: bool) -> BaseModelInterface:
    return ModelFactory.create(model_type, interface=BaseModelAdapter)(
        model_type=model_type,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
    )


def main(args):
    config_dir = get_practice_dir()
    practice_manager = PracticeManager(official_config_dir=config_dir)
    dataset_dir = get_dataset_dir()
    dataset_loader = FileDatasetLoader(dataset_dir)
    quant_service = QuantServiceProxy(dataset_loader)

    app = NaiveQuantizationApplication(
        practice_manager=practice_manager,
        quant_service=quant_service,
        model_factory=create_model,
    )

    app.quant(
        model_type=args.model_type,
        model_path=args.model_path,
        save_path=args.save_path,
        device=args.device,
        quant_type=args.quant_type,
        config_path=args.config_path,
        trust_remote_code=args.trust_remote_code
    )
