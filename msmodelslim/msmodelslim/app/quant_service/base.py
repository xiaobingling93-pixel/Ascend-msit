#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any

from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from msmodelslim.utils.exception import ToDoError
from .. import DeviceType
from ..base.quant_config import BaseQuantConfig


class BaseQuantService(ABC):
    backend_name: str = "Unknown"

    def __init__(self, dataset_loader: DatasetLoaderInterface):
        self.dataset_loader = dataset_loader

    @abstractmethod
    def quantize(
            self,
            quant_config: BaseQuantConfig,
            model_adapter: Any,
            save_path: Optional[Path] = None,
            device: DeviceType = DeviceType.NPU,
    ) -> None:
        raise ToDoError("quantize is not implemented",
                        action="Please implement quantize for your quant service")
