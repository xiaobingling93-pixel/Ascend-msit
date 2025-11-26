# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Type, TypeVar, Optional

from msmodelslim.utils.exception import ToDoError, UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.model.interface import IModel, IModelFactory


MODEL_ADAPTER_ENTRY_POINTS = "msmodelslim.model_adapter.plugins"

DEFAULT = "default"


class PluginModelFactory(IModelFactory):

    _model_map = None

    @classmethod
    def create(
        cls,
        model_type: str,
        model_path: Path,
        trust_remote_code: bool = False,
    ) -> IModel:
        model_map = PluginModelFactory._get_model_map()

        if model_type not in model_map:
            if DEFAULT in model_map:
                get_logger().warning(
                    f"Model adapter '{model_type}' not found, trying default adapter..."
                )
                model_type = DEFAULT
            else:
                raise UnsupportedError(
                    f"No adapter found for '{model_type}' and no default adapter registered. "
                    f"Registered adapters: {list(model_map.keys())}"
                )

        adapter_class = model_map[model_type].load()

        adapter_instance = adapter_class(
            model_type=model_type,
            model_path=model_path,
            trust_remote_code=trust_remote_code,
        )
        return adapter_instance

    @classmethod
    def _get_model_map(cls) -> dict:
        if cls._model_map is None:
            if sys.version_info >= (3, 10):
                eps = entry_points().select(group=MODEL_ADAPTER_ENTRY_POINTS)
            else:
                eps = entry_points().get(MODEL_ADAPTER_ENTRY_POINTS, [])
            cls._model_map = {ep.name: ep for ep in eps}
            get_logger().info(
                f"Found {len(cls._model_map)} model adapters: "
                f"{list(cls._model_map.keys())}"
            )
        return cls._model_map
