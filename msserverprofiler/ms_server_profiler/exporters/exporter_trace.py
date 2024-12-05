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

from ms_server_profiler.parse import ExporterBase


class ExporterTrace(ExporterBase):
    name = "trace"

    @classmethod
    def intialize(cls, args):
        cls.args = args

    @classmethod
    def export(cls, data) -> None:
        from parse_data_to_trace import create_trace_events, save_trace_data_into_json
        all_data_df, cpu_data_df = data['tx_data_df'], data['cpu_data_df']
        output = cls.args.output_path
        trace_data = create_trace_events(all_data_df, cpu_data_df)
        save_trace_data_into_json(trace_data, output)
