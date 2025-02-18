# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from abc import abstractmethod, ABC


class BaseSaver(ABC):

    @abstractmethod
    def pre_process(self) -> None:
        ...

    @abstractmethod
    def save(self, name, meta, data) -> None:
        ...

    @abstractmethod
    def post_process(self) -> None:
        ...
