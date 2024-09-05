from abc import ABC, abstractmethod
import torch 


class DumpFileReader(ABC):
    def __init__(self, path: str):
        self.path = path 
        self.key_to_folder = self._map_keys_to_folders()

    @abstractmethod
    def _map_keys_to_folders(self) -> dict:
        pass

    @abstractmethod
    def _get_keys(self) -> set:
        pass

    @abstractmethod
    def get_tensor(self, key: str) -> torch.Tensor:
        pass 
