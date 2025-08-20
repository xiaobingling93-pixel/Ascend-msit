# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from pathlib import Path

from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from msmodelslim.app.base.const import DeviceType


class BaseModelAdapter(ABC):
    def __init__(self, model_type: str, ori_path: Path, device: DeviceType = DeviceType.NPU, trust_remote_code=False):
        self._type = model_type
        self._ori_path = ori_path
        self._device = device
        self._path = None

        self._pedigree = self._get_model_pedigree()
        self._config = self._load_config()
        self._tokenizer = self._load_tokenizer(trust_remote_code=trust_remote_code)
        self._model = self._load_model(device=device, trust_remote_code=trust_remote_code)
        self._load_hook()

    @property
    def ori(self) -> Path:
        """
        The original path of the model.
        """
        return self._ori_path

    @property
    def path(self) -> Path:
        """
        The path of the quantized model.
        """
        if self._path is None:
            raise ValueError("Model is not persisted. Please call persisted() to persist the model.")
        return self._path

    @property
    def device(self) -> DeviceType:
        """
        The device of the model.
        """
        return self._device

    @property
    def config(self) -> PretrainedConfig:
        """
        The config of the model.
        """
        return self._config

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """
        The tokenizer of the model.
        """
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """
        The model of the model.
        """
        return self._model

    @property
    def type(self) -> str:
        """
        The type of the model.
        """
        return self._type

    @property
    def pedigree(self) -> str:
        """
        The pedigree of the model.
        """
        return self._pedigree

    @abstractmethod
    def _get_model_pedigree(self) -> str:
        """
        Set the pedigree of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_config(self) -> PretrainedConfig:
        """
        Load the config of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        """
        Load the tokenizer of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_model(self, device: DeviceType = DeviceType.NPU, trust_remote_code: bool = False) -> PreTrainedModel:
        """
        Load the model of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_hook(self) -> None:
        """
        Run after the model is loaded.
        """
        pass

    @abstractmethod
    def _persist_hook(self) -> None:
        """
        Run after the model is persisted.
        """
        pass

    def persisted(self, save_path: Path) -> None:
        if not isinstance(save_path, Path):
            raise ValueError("save_path must be a Path object")
        if not save_path.exists():
            raise ValueError("save_path must exist")
        self._path = save_path
        self._persist_hook()
