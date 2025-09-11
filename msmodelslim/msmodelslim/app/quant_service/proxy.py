#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

from pathlib import Path
from typing import Optional, Any

from msmodelslim.app import DeviceType
from msmodelslim.app.base import BaseQuantConfig
from msmodelslim.app.quant_service import BaseQuantService, DatasetLoaderInterface, load_plugins, load_quant_service_cls
from msmodelslim.utils.logging import logger_setter


@logger_setter(prefix='msmodelslim.app.quant_service.proxy')
class QuantServiceProxy(BaseQuantService):

    def __init__(self, dataset_loader: DatasetLoaderInterface):
        super().__init__(dataset_loader)
        self.quant_service: Optional[BaseQuantService] = None

    def quantize(
            self,
            quant_config: BaseQuantConfig,
            model_adapter: Any,
            save_path: Optional[Path] = None,
            device: DeviceType = DeviceType.NPU,
    ) -> None:
        load_plugins()
        self.quant_service = load_quant_service_cls(quant_config.apiversion)(self.dataset_loader)
        self.quant_service.quantize(
            quant_config=quant_config,
            model_adapter=model_adapter,
            save_path=save_path,
            device=device,
        )
