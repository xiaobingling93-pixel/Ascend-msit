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

from ms_server_profiler.utils import convert_syscnt_to_ts
from ms_server_profiler.parse import PluginBase


class PluginCpuTimeStamp(PluginBase):
    name = "plugin_cpu_timestamp"
    depends = []

    @classmethod
    def parse(cls, data):
        cpu_data_df = data.get('cpu_data_df')
        cpu_start_cnt = data.get('cpu_start_cnt')
        cpu_frequency = data.get('cpu_frequency')

        cpu_data_df['start_time'] = convert_syscnt_to_ts(cpu_data_df['start_time'], cpu_start_cnt, cpu_frequency)
        cpu_data_df['end_time'] = convert_syscnt_to_ts(cpu_data_df['end_time'], cpu_start_cnt, cpu_frequency)
        data['cpu_data_df'] = cpu_data_df
        return data
