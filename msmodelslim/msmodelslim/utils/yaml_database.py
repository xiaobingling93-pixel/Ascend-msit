# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from pathlib import Path
from typing import Dict, Any

import yaml


class YamlDatabase:
    def __init__(self, config_dir: Path):
        """Load all configs from modelType subdirectories"""
        self.config_by_name: Dict[str, Any] = {}     # [config_id] -> ConfigType
        self.config_dir = config_dir

        if not self.config_dir.is_dir():
            raise ValueError(f"YamlDatabase need a directory, {self.config_dir} is not a directory")

    def load_config(self):
        """Load configuration from a YAML file"""
        for config_file in self.config_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                yield yaml.safe_load_all(f)
