# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from pathlib import Path
from typing import List

from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from msmodelslim.utils.exception import InvalidDatasetError, SchemaValidateError
from msmodelslim.utils.security import get_valid_read_path
from msmodelslim.utils.security.model import SafeGenerator


class FileDatasetLoader(DatasetLoaderInterface):
    def __init__(self, dataset_dir: Path):
        self.dir = dataset_dir

    def get_dataset_by_name(self, dataset_id: str) -> List[str]:
        if not isinstance(dataset_id, str):
            raise SchemaValidateError(f'dataset_id must be a str',
                                      action='Please make sure the dataset_id is a string')
        anti_path = self.dir / dataset_id
        get_valid_read_path(str(anti_path), "jsonl")
        try:
            return SafeGenerator.load_jsonl(anti_path)
        except Exception as e:
            raise InvalidDatasetError(f'Failed to load dataset {dataset_id}',
                                      action='Please ensure the dataset exists and is valid') from e
