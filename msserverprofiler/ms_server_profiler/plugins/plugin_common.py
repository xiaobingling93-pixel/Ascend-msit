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

from enum import Enum
import json
from ms_server_profiler.parse import PluginBase
from ms_server_profiler.utils import convert_syscnt_to_ts



class PluginCommon(PluginBase):
    name = "plugin_common"
    depends = []

    @classmethod
    def parse(cls, data):
        all_data_df = data["tx_data_df"]
        sys_start_cnt = data["sys_start_cnt"]
        cpu_frequency = data["cpu_frequency"]

        all_data_df = data_convert(all_data_df, sys_start_cnt, cpu_frequency)
        data["tx_data_df"] = all_data_df
        return data
    

class ReqStatus(Enum):
    WAITING = 0
    PENDING = 1
    RUNNING = 2
    SWAPPED = 3
    RECOMPUTE = 4
    SUSPENDED = 5
    END = 6
    STOP = 7
    PREFILL_HOLD = 8


def extract_ids_from_reslist(message, rid_map):
    res_list = message.get('resList', None)
    rid = []
    token_id = []
    if res_list:
        for req in res_list:
            rid.append(rid_map.get(req.get('rid', None), req.get('rid', None)))
            token_id.append(req.get('iter', None))
        if len(res_list) > 1:
            return ','.join(rid), ','.join(token_id)
        elif rid and token_id:
            return str(rid[0]), str(token_id[0])
    return res_list, res_list


def convert_message_to_json(message):
    if message.startswith('{') and message.endswith('}'):
        return json.loads(message)
    else:
        message = '{' + message[:-1] + '}'
        return json.loads(message)
    

def find_during_time_by_span_id(all_data_df):
    all_data_df['during_time'] = all_data_df['end_time'] - all_data_df['start_time']
    return all_data_df


def extract_batch_type(message, rid_map):
    _, token_id = extract_ids_from_reslist(message, rid_map)
    if token_id is None:
        return token_id
    token_list = token_id.split(',') if isinstance(token_id, str) else [token_id]
    if all(token == '0' for token in token_list):
        return 'Prefill'
    elif '0' in token_list and len(set(token_list)) > 1:
        return 'Prefill, Decode'
    else:
        return 'Decode'


def get_state_name_by_value(value):
    if value is not None:
        try:
            return ReqStatus(value).name
        except ValueError:
            return str(value)
    else:
        raise ValueError("Failed to get ReqState since new_value is None.")
    

def extract_rid(message, rid_map):
    rid_from_message = message.get('rid', None)
    if rid_from_message is not None:
        return str(rid_map.get(rid_from_message, rid_from_message))

    rid_from_reslist, _ = extract_ids_from_reslist(message, rid_map)
    if rid_from_reslist is not None:
        return rid_from_reslist
    else:
        return rid_from_message

def data_convert(all_data_df, sys_start_cnt, cpu_frequency):
    all_data_df['start_time'] = convert_syscnt_to_ts(all_data_df['start_time'], sys_start_cnt, cpu_frequency)
    all_data_df['end_time'] = convert_syscnt_to_ts(all_data_df['end_time'], sys_start_cnt, cpu_frequency)
    all_data_df['message'] = all_data_df['message'].apply(lambda x: convert_message_to_json(x))
    all_data_df['type'] = all_data_df['message'].apply(lambda x: x.get("type"))
    rid_link_map = {x.get("from"): x.get("to") for x in all_data_df[all_data_df["type"] == 3]["message"]}
    all_data_df['rid'] = all_data_df['message'].apply(lambda x: extract_rid(x, rid_link_map))
    all_data_df['batch_type'] = all_data_df['message'].apply(lambda x: extract_batch_type(x, rid_link_map))
    all_data_df['name'] = all_data_df['message'].apply(lambda x: x.get('name', None))
    all_data_df.loc[all_data_df['name'] == 'ReqState', 'name'] = (
        all_data_df.loc[all_data_df['name'] == 'ReqState', 'message'].apply(
            lambda x: get_state_name_by_value(x.get('new_value', None))))
    all_data_df = find_during_time_by_span_id(all_data_df)
    return all_data_df


