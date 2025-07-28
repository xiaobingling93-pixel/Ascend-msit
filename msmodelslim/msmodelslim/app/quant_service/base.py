# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from ..base.model import BaseModel
from ..base.quant_config import BaseQuantConfig


class BaseQuantService(ABC):
    def __init__(self, dataset_loader: DatasetLoaderInterface):
        self.dataset_loader = dataset_loader

    @abstractmethod
    def quantize(self, model: BaseModel, quant_config: BaseQuantConfig, save_path: Optional[Path] = None) -> None:
        raise NotImplementedError
