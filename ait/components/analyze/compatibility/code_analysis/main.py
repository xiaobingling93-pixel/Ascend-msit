# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import pathlib
import click

import utils
import model

opt_path = click.argument(
    'path', nargs=1, type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=pathlib.Path)
)


@click.command()
@opt_path
def code_analysis(path):
    model.evaluate(path)


if __name__ == '__main__':
    code_analysis()
