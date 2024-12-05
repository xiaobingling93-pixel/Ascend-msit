# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import argparse

from ms_server_profiler.plugins.plugin_common import PluginCommon
from ms_server_profiler.plugins.plugin_cpu_timestamp import PluginCpuTimeStamp
from ms_server_profiler.exporters.exporter_trace import ExporterTrace
from ms_server_profiler_analyze.plugins.plugin_req_status import PluginReqStatus
from ms_server_profiler_analyze.exporters.exporter_req_status import ExporterReqStatus
from ms_server_profiler.parse import parse


def init_exporters(exporter_classes, args):
    exporter_class_map = {exporter_cls.name: exporter_cls for exporter_cls in exporter_classes}
    selected_exporter = []
    for name in args.exporter:           
        exporter = exporter_class_map.get(name)
        if exporter is None:
            continue
        exporter.initialize(args)
        selected_exporter.append(exporter)

    from pathlib import Path
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    return selected_exporter



def check_input_path_valid(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Path is not a valid directory: {path}")
    if '..' in path:
        raise argparse.ArgumentTypeError(f"Path contains illegal characters: {path}")
    return path


def check_output_path_valid(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.access(path, os.W_OK):
        raise argparse.ArgumentTypeError(f"Output path is not writable: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='MS Server Profiler')
    parser.add_argument(
        '--input_path',
        type=check_input_path_valid,
        help='Path to the folder containing profile data.')
    parser.add_argument(
        '--output_path',
        type=check_output_path_valid,
        default=os.getcwd(),
        help='Output file path to save results.')
    parser.add_argument(
        '--exporter',
        type=str,
        nargs='+',
        default=['trace', 'req_status'],
        help='exporter to use')

    args = parser.parse_args()
    plugins = [PluginCommon, PluginReqStatus, PluginCpuTimeStamp]
    exporter_classes = [ExporterTrace, ExporterReqStatus]

    exporters = init_exporters(exporter_classes, args)
    parse(args.input_path, plugins, exporters)


if __name__ == '__main__':
    main()

