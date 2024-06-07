# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import pathlib
from typing import Dict, List
import click

from src import analysis
from src.knowledge import Knowledge

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def print_result(result: Dict[Knowledge, List[str]]):
    if len(result) == 0:
        return
    logger.info()
    logger.info('============= Analysis Result =============')
    logger.info()
    for knowledge, match_infos in result.items():
        logger.info(f'{knowledge.suggestion}')
        logger.info('查询和匹配到的接口、文件路径和行号如下：')
        for match_info in match_infos:
            logger.info(f'  {match_info}')
        logger.info()


opt_path = click.argument(
    'path', nargs=1, type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=pathlib.Path)
)

opt_scene = click.option(
    '-s', '--scene', 'scene', default='310->310B', type=str, help='scene you want to analysis, default 310->310B.'
)


@click.command()
@opt_path
@opt_scene
def analysis_acl_api(path, scene):
    """analysis application code and print suggestions"""
    if scene != '310->310B':
        logger.info(f'[error] not support scene: {scene}.')
        return

    result = analysis.analysis_310_to_310b(path)
    print_result(result)


if __name__ == '__main__':
    analysis_acl_api()
