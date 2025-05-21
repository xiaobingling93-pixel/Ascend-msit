#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Dict, Any, Optional, Union, Callable
import contextvars

from msmodelslim import logger


class TimestepManager:
    """Manages timestep indices for multi-modal quantization processes."""

    _timestep_var = contextvars.ContextVar("timestep_idx", default=None)

    @classmethod
    def set_timestep_idx(cls, t_idx: int) -> None:
        """Set the current timestep index."""
        current = cls._timestep_var.get()
        if current is not None and current == t_idx:
            logger.debug("Warning: Setting same timestep value consecutively: %r", t_idx)
        cls._timestep_var.set(t_idx)
        logger.debug("Timestep index set to: %r", t_idx)

    @classmethod
    def get_timestep_idx(cls) -> Optional[int]:
        """Get the current timestep index."""
        t_idx = cls._timestep_var.get()
        if t_idx is None:
            logger.debug("Warning: Timestep index not set. Call set_timestep_idx() before each timestep.")
        return t_idx


