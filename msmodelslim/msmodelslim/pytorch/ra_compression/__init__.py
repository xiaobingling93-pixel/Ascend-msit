# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
__all__ = ['RACompressConfig']

from msmodelslim.pytorch.ra_compression.ra_config import RACompressConfig

try:
    from msmodelslim.pytorch.ra_compression.ra_tools import RACompressor
except ModuleNotFoundError as exception:
    from msmodelslim import logger

    logger.info("can not import RACompressor")
else:
    __all__ += ['RACompressor']