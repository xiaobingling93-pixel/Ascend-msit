import os
from abc import ABC, abstractmethod
from typing import Optional, Union, Mapping, Type, Dict

import torch
from typing_extensions import Self

from msmodelslim.pytorch.llm_ptq.accelerate_adapter import get_state_dict_copy
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import WritableOffloadedWeightsLoader, \
    get_offloaded_weights_loader_if_have
from msmodelslim import logger as msmodelslim_logger

OFFLOAD_MEMORY = "memory"
OFFLOAD_DISK = "disk"


class StateDictConfig(ABC):
    typ = None
    skip_keys = None

    @property
    @abstractmethod
    def args(self) -> Mapping:
        raise NotImplementedError(
            f"config [{type(self).__name__}] is missing the required \"args\" method")


class StateDictBase(Mapping, ABC):
    @classmethod
    @abstractmethod
    def from_model(cls, model: torch.nn.Module, config: StateDictConfig) -> Self:
        raise NotImplementedError(
            f"State Dict [{cls.__name__}] is missing the required \"from_model\" method")


def copy_offloaded_state_dict(model: torch.nn.Module, config: StateDictConfig) -> Mapping:
    if not isinstance(config, StateDictConfig):
        raise ValueError("state dict config must be StateDictConfig")

    return _select_state_dict(config.typ).from_model(model, config)


class MemoryStateDictConfig(StateDictConfig):
    typ = OFFLOAD_MEMORY

    def args(self) -> Mapping:
        return {}


class MemoryStateDict(dict, StateDictBase):
    """
    输入存入内存的state_dict
    """

    @classmethod
    def from_model(cls, model: torch.nn.Module, config: MemoryStateDictConfig):
        return MemoryStateDict(get_state_dict_copy(model, skip_keys=config.skip_keys))


class DiskStateDictConfig(StateDictConfig):
    typ = OFFLOAD_DISK
    ARG_SAVE_FOLDER = "save_folder"

    def __init__(self) -> None:
        super().__init__()
        self.__save_folder = None

    def save_folder(self, __save_folder: Optional[Union[str, os.PathLike]]) -> Self:
        if not isinstance(__save_folder, (str, os.PathLike)):
            raise ValueError("path to state dict must be str or os.PathLike")

        self.__save_folder = __save_folder
        return self

    @property
    def args(self):
        return {self.ARG_SAVE_FOLDER: self.__save_folder}


class DiskStateDict(WritableOffloadedWeightsLoader, StateDictBase):
    """
    数据存入磁盘的state_dict
    """

    logger = msmodelslim_logger

    def __init__(self, save_folder: Optional[Union[str, os.PathLike]]):
        os.makedirs(save_folder, exist_ok=True)
        super().__init__(save_folder=save_folder, index={})

    @classmethod
    def from_model(cls, model: torch.nn.Module, config: DiskStateDictConfig):
        state_dict = cls(**config.args)
        weights_loader = get_offloaded_weights_loader_if_have(model)
        for key, value in model.state_dict().items():
            cls.logger.debug(f"{key} is on {value.device.type}")
            if config.skip_keys is not None and key in config.skip_keys:
                state_dict[key] = torch.zeros([1])
            elif value.device.type == 'meta':
                # load from model's offload weights loader
                state_dict[key] = weights_loader[key].clone().detach().cpu()
            else:
                state_dict[key] = value.clone().detach().cpu()
        return state_dict


_default_state_dict = MemoryStateDict

_state_dict_class_map: Dict[str, StateDictBase] = {
    OFFLOAD_MEMORY: MemoryStateDict,
    OFFLOAD_DISK: DiskStateDict,
}


def _select_state_dict(typ: str) -> Type[StateDictBase]:
    return _state_dict_class_map.get(typ, _default_state_dict)
