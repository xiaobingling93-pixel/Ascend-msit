# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseCollector(ABC):
    @abstractmethod
    def collect(self) -> dict:
        pass


class ConfigCollector(BaseCollector):
    PATH_SEPARATOR = '.'

    def __init__(self, config_path: str):
        self.config_path = config_path

    @staticmethod
    def load_json(config_path):
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e

    @staticmethod
    def flatten_dict_leaves(result, data, parent_path="", sep=PATH_SEPARATOR):
        """Recursively flattens a nested dict/list structure into a dict with path keys."""
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{parent_path}{sep}{key}" if parent_path else key
                ConfigCollector.flatten_dict_leaves(result, value, path, sep)
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                path = f"{parent_path}[{idx}]" if parent_path else f"[{idx}]"
                ConfigCollector.flatten_dict_leaves(result, value, path, sep)
        else:
            result[parent_path] = data

    def collect(self):
        json_data = self.load_json(self.config_path)
        result = {}
        self.flatten_dict_leaves(result, json_data)
        return result


class ParallelCollector(BaseCollector):
    """Base class for parallel sub-collectors."""

    def __init__(self, sub_collectors):
        self.sub_collectors = sub_collectors

    def collect(self):
        if not isinstance(self.sub_collectors, (dict, list)):
            raise TypeError("sub_collectors must be a dict or a list")

        is_dict = isinstance(self.sub_collectors, dict)
        collectors = (
            self.sub_collectors.items() if is_dict else enumerate(self.sub_collectors)
        )
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.sub_collectors)) as executor:
            future_to_key = {
                executor.submit(collector.collect): key
                for key, collector in collectors
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"error": str(e)}
                if is_dict:
                    results[key] = result
                else:
                    results.update(result)
        return results
