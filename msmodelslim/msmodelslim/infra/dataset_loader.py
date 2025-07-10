# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from pathlib import Path
from typing import List

from ascend_utils.common.security import get_valid_read_path
from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from msmodelslim.utils.safe_utils import SafeGenerator


class FileDatasetLoader(DatasetLoaderInterface):
    def __init__(self, dataset_dir: Path):
        self.dir = dataset_dir

    def get_dataset_by_name(self, dataset_id: str) -> List[str]:
        if not isinstance(dataset_id, str):
            raise ValueError(f'dataset_id must be a str')
        anti_path = self.dir / dataset_id
        get_valid_read_path(str(anti_path), "jsonl")
        return SafeGenerator.load_jsonl(anti_path)
