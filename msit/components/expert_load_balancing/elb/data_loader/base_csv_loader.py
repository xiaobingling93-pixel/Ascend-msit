# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import pandas as pd

from components.expert_load_balancing.elb.data_loader.base_loader import BaseDataLoader
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
    