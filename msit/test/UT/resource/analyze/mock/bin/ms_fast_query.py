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
import os
import shutil

import click


opt_type = click.option(
    '-t', '--type', 'query_type', type=click.Choice(['op', 'model']), required=True, help='fast query type.'
)

opt_opp_path = click.option('--opp_path', 'opp_path', type=str, help='opp path, required when type is op')

opt_out = click.option('-o', '--output', 'output', type=str, required=True, help='output file path')


@click.command()
@opt_type
@opt_opp_path
@opt_out
def fast_query(query_type, opp_path, output) -> None:
    cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    data_path = os.path.join(cur_dir, 'resource', 'analyze', 'dataset', 'opp', 'opp.json')

    out_path = os.path.realpath(output)
    if data_path == out_path:
        return

    shutil.copyfile(data_path, out_path)


if __name__ == '__main__':
    fast_query()
