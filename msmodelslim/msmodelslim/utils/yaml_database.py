# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from pathlib import Path

from ascend_utils.common.security import yaml_safe_dump
from ascend_utils.common.security.path import get_write_directory, get_valid_read_path, \
    yaml_safe_load


class YamlDatabase:
    def __init__(self, config_dir: Path, read_only: bool = True):
        """Load all configs from modelType subdirectories"""
        if read_only:
            get_valid_read_path(str(config_dir), is_dir=True)
        else:
            get_write_directory(str(config_dir), write_mode=0o750)
        self.config_dir = config_dir
        self.read_only = read_only

    def __iter__(self):
        return (config_file.stem for config_file in self.config_dir.glob("*.yaml"))

    def __getitem__(self, item):
        """Load value from a YAML file"""
        if not isinstance(item, str):
            raise TypeError(f"yaml database key must be a string, but got {type(item)}")

        try:
            value_file = self.config_dir / f"{item}.yaml"
            return yaml_safe_load(str(value_file))
        except FileNotFoundError as e:
            raise KeyError(f"yaml database key {item} not found") from e

    def __setitem__(self, key, value):
        """Save value to a YAML file"""
        if self.read_only:
            raise ValueError(f"yaml database {self.config_dir} is read-only")

        if not isinstance(key, str):
            raise TypeError(f"yaml database key must be a string, but got {type(key)}")

        value_file = self.config_dir / f"{key}.yaml"
        yaml_safe_dump(value, str(value_file))

    def __contains__(self, item):
        """Check if a YAML file exists"""
        return (self.config_dir / f"{item}.yaml").exists()

    def values(self):
        """Load values from a YAML directory"""
        for key in self:
            yield self[key]
