# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import Optional, List, Generator


class NaiveQuantizationInterface(ABC):
    @abstractmethod
    def get_task_by_name(self, model_type, config_id: str) -> Optional[dict]:
        """Get configuration by ID"""
        pass

    @abstractmethod
    def get_task_by_path(self, config_path: str) -> Optional[dict]:
        """Get configuration by path"""
        pass
    
    @abstractmethod
    def iter_task(self, model_type) -> Generator[dict, None, None]:
        """Iterate configurations by priority"""
        pass
