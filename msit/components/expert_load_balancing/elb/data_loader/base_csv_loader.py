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
import pandas as pd

from components.expert_load_balancing.elb.data_loader.base_loader import DataType, BaseDataLoader
from components.utils.security_check import get_valid_read_path
from components.utils.log import logger


class BaseCsvLoader(BaseDataLoader):
    def __init__(self, input_path):
        super().__init__(input_path)

    @staticmethod
    def load_from_file(csv_path):
        csv_path = get_valid_read_path(csv_path)
        try:
            df = pd.read_csv(csv_path, header=None)
            res = df.to_numpy()
            if len(res.shape) != 2:
                logger.warning(f"Data in csv shape is illegal.")
                return None
            return res
        except Exception:
            return None

    @staticmethod
    def load_with_bak_file(csv_path, bak_path=""):
        res = BaseCsvLoader.load_from_file(csv_path=csv_path)
        if res is not None:
            return res
        if bak_path:
            logger.warning(f"Load from file: {csv_path} failed.")
            logger.info(f"Try load from bak file: {bak_path}")
            res = BaseCsvLoader.load_from_file(csv_path=bak_path)
            if res is not None:
                return res
                
        logger.error(f"Both file and bak file is not readable.")
        raise RuntimeError("Load from file failed.")
    