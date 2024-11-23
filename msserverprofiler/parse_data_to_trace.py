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

import sqlite3
import json
import os
import stat
from datetime import datetime, timezone
import argparse
from enum import Enum

import pandas as pd
import psutil


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


DEFAULT_FREQ = 1000000000
FREQ_TO_100MHZ = 100 * 1000 * 1000
NANO_SECOND = 1000 * 1000 * 1000
SYS_TS = psutil.boot_time()


def find_config_files(folder_path):
    config_path = None
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename == 'host_start.log':
                config_path = os.path.join(root, filename)
    if config_path is None:
        raise ValueError(f"No valid config files found in {folder_path}, please check.")
    return config_path


def get_sys_start_cnt(folder_path):
    sys_start_cnt = 0
    config_path = find_config_files(folder_path)
    with open(config_path, 'r') as f:
        for line in f.readlines():
            if "cntvct:" in line:
                key, value = line.strip().split(": ")
                if key == "cntvct":
                    sys_start_cnt = int(value)
    return sys_start_cnt


def load_data_from_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM MsprofTxEx
    """)

    all_data = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    all_data_df = pd.DataFrame(all_data, columns=columns)
    conn.close()
    return all_data_df


def concat_data_from_folder(folder_path):
    full_df = pd.DataFrame()
    
    def merge_message(series):
        series_merge = series.sort_values("mark_id")
        series_merge.iloc[0, series_merge.columns.get_loc("message")] = "".join(series_merge["message"])
        return series_merge.iloc[0]
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename == 'msproftx.db':
                db_path = os.path.join(root, filename)
                data_df = load_data_from_database(db_path)
                
                data_df[["span_id", "message"]] = (data_df[["mark_id", "message"]].apply(
                    lambda x: pd.Series(extract_span_info_from_message(x["message"], x["mark_id"])), axis=1
                ))
                data_df = data_df.groupby("span_id").apply(merge_message, include_groups=True)
                
                full_df = pd.concat([full_df, data_df], ignore_index=True)
    if full_df.empty:
        raise ValueError(f"No valid database found in {folder_path}, please check.")
    full_df = full_df.sort_values(by='start_time', ascending=True).reset_index(drop=True)
    return full_df


def convert_nano_to_ts(cnt):
    return (SYS_TS + ((cnt) / NANO_SECOND)) * 1000 * 1000


def convert_cntvct_to_ts(cnt):
    return (SYS_TS + ((cnt) / FREQ_TO_100MHZ)) * 1000 * 1000


def convert_syscnt_to_ts(cnt, sys_start_cnt):
    return (SYS_TS + ((cnt - sys_start_cnt) / DEFAULT_FREQ)) * 1000


def extract_span_info_from_message(message, mark_id):
    span_id = str(mark_id)
    if message.startswith('span') and '|' in message and '=' in message:
        span_part = message.split("|")[0]
        span_id = span_part.split("=")[1]
        message = message.split("|")[1]
        return span_id, message
    return span_id, message


def convert_message_to_json(message):
    if message.startswith('{') and message.endswith('}'):
        return json.loads(message)
    else:
        message = '{' + message[:-1] + '}'
        return json.loads(message)


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


def extract_rid(message, rid_map):
    rid_from_message = message.get('rid', None)
    if rid_from_message is not None:
        return str(rid_map.get(rid_from_message, rid_from_message))

    rid_from_reslist, _ = extract_ids_from_reslist(message, rid_map)
    if rid_from_reslist is not None:
        return rid_from_reslist
    else:
        return rid_from_message


def extract_batch_type(message, rid_map):
    _, token_id = extract_ids_from_reslist(message, rid_map)
    if token_id is None:
        return token_id
    token_list = token_id.split(',') if isinstance(token_id, str) else [token_id]
    if len(token_list) == 1 and token_list[0] == '0':
        return 'Prefill'
    elif '0' in token_list:
        return 'Prefill, Decode'
    else:
        return 'Decode'


def modify_rid(rid):
    if rid is not None:
        if rid.startswith('endpoint_common_'):
            rid_num = int(rid[len('endpoint_common_'):])
            return str(rid_num - 1)
    return rid


def get_state_name_by_value(value):
    if value:
        return ReqStatus(value).name
    else:
        return value


def find_during_time_by_span_id(all_data_df):
    span_with_dur_time = {}
    all_data_df['during_time'] = all_data_df['end_time'] - all_data_df['start_time']
    return all_data_df


def data_convert(all_data_df, sys_start_cnt):
    all_data_df['start_time'] = convert_cntvct_to_ts(all_data_df['start_time'])
    all_data_df['end_time'] = convert_cntvct_to_ts(all_data_df['end_time'])
    all_data_df['message'] = all_data_df['message'].apply(lambda x: convert_message_to_json(x))
    all_data_df['type'] = all_data_df['message'].apply(lambda x: x.get("type"))
    rid_link_map = {x.get("from"): x.get("to") for x in all_data_df[all_data_df["type"] == 3]["message"]}
    all_data_df['rid'] = all_data_df['message', 'mark_id'].apply(lambda x: extract_rid(x["message"], rid_link_map), axis=1)
    all_data_df['batch_type'] = all_data_df['message'].apply(lambda x: extract_batch_type(x, rid_link_map))
    all_data_df['rid'] = all_data_df['rid'].apply(lambda x: modify_rid(x))
    all_data_df['name'] = all_data_df['message'].apply(lambda x: x.get('name', None))
    all_data_df.loc[all_data_df['name'] == 'ReqState', 'name'] = (
        all_data_df.loc[all_data_df['name'] == 'ReqState', 'message'].apply(
            lambda x: get_state_name_by_value(x.get('new_value', None))))
    all_data_df = find_during_time_by_span_id(all_data_df)
    return all_data_df


def add_args_for_state_type(message):
    args = {}
    name = message.get('name', None)
    if name == 'httpReq':
        args['recvTokenSize'] = message.get('recvTokenSize', None)
    if name == 'ReqEnQueue':
        args['queueID'] = message.get('queue', None)
        args['queueSize'] = message.get('size', None)
    if name == 'deviceKvCache':
        args['kvCacheValue'] = message.get('value', None)
        args['name'] = message.get('event', None)
    if name == 'hostKvCache':
        args['kvCacheValue'] = message.get('value', None)
        args['name'] = message.get('event', None)
    if name == 'ReqDeQueue':
        args['queueID'] = message.get('queue', None)
        args['queueSize'] = message.get('size', None)
    if name == 'httpRes':
        args['replyTokenSize'] = message.get('replyTokenSize', None)
    return args


def create_trace_events(all_data_df, cpu_data_df):
    trace_events = []

    for idx, data in all_data_df.iterrows():
        if data['event_type'] == 'marker' and data['name'] is not None:
            trace_events.append(
                {
                    "name": data['name'],
                    "ph": "I",
                    "ts": data['start_time'],
                    "pid": data['pid'],
                    "tid": data['name'],
                    "cat": "Request Status",
                    "args": {**{'rid': data['rid']}, **add_args_for_state_type(data['message'])}
                },
            )
        if data['event_type'] == "start/end" and data['name'] is not None:
            trace_events.append(
                {
                    "name": data['name'],
                    "ph": "X",
                    "ts": data['start_time'],
                    "dur": data['during_time'],
                    "pid": data['pid'],
                    "tid": data['name'],
                    "cat": "Execute",
                    "args": {
                        'rid': data['rid'],
                        'resList': data['message'].get('resList', None),
                        'batchType': data['batch_type']
                    }
                },
            )
        if data['rid'] is not None:
            rids = str(data["rid"]).split(",")
            for rid in rids:
                flow_event = {
                        "name": "flow_" + rid,
                        "id": rid,
                        "cat": rid,
                        "pid": data['pid'],
                        "tid": data['name'],
                        "ts": data['start_time']
                    }
                if data["name"] == "httpReq":
                    flow_event["ph"] = 's'
                elif data["name"] == "httpRes":
                    flow_event["ph"] = 'f'
                else:
                    flow_event["ph"] = 't'
                trace_events.append(flow_event)
        if data['type'] == 1:
            trace_events.append(
                {
                    "name": data["name"],
                    "ph": "C",
                    "ts": data['start_time'],
                    "pid": data['pid'],
                    "tid": "NPU Usage",
                    "cat": "Metrics",
                    "args": {
                        'NPU Usage': data['message'].get('value', None)
                    }
                }
            )

    trace_events = add_cpu_events(cpu_data_df, trace_events)
    trace_events = sort_trace_events_by_cat(trace_events)

    trace_data = {"traceEvents": trace_events}
    return trace_data


def sort_trace_events_by_cat(trace_events):
    sorting_order = ['Metrics', 'Request Status', 'Execute']

    def get_sorting_key(event):
        if 'cat' in event and event["cat"] in sorting_order:
            return sorting_order.index(event['cat'])
        else:
            return float('inf')

    sort_events_by_cat = sorted(
        (event for event in trace_events if 'cat' in event),
        key=get_sorting_key
    )
    event_without_cat = [event for event in trace_events if 'cat' not in event]
    
    tid_sorting_order = ['deviceKvCache', 'hostKvCache', 'httpReq', 'httpRes', 'ReqEeQueue', 'ReqDeQueue', 'ReqState', 'BatchSchedule']
    
    main_pid = 0
    for event_info in trace_events:
        if event_info.get("name") in tid_sorting_order:
            main_pid = event_info.get("pid")
            break
    tid_sorting_meta = [dict(
        name="thread_sort_index",
        ph="M",
        pid=main_pid,
        tid=tid,
        args=dict(sort_index=index)) for index, tid in enumerate(tid_sorting_order)]
        
    sorted_trace_events = sort_events_by_cat + event_without_cat + tid_sorting_meta
    return sorted_trace_events


def add_cpu_events(cpu_data_df, trace_events):
    for _, data in cpu_data_df.iterrows():
        trace_events.append(
            {
                "name": "CPU Usage",
                "ph": "C",
                "ts": data['start_time'],
                "pid": 1,
                "tid": "CPU Usage",
                "cat": "Metrics",
                "args": {
                    'CPU Usage': data['usage']
                }
            }
        )
    return trace_events


def save_trace_data_into_json(trace_data, output):
    current_datetime = datetime.now(tz=timezone.utc)
    datetime_str = current_datetime.strftime("%Y%m%d%H%M%S")

    file_path = os.path.join(output, f'chrome_tracing_{datetime_str}.json')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    file_descriptor = os.open(file_path, flags, mode)
    with os.fdopen(file_descriptor, 'w') as f:
        json.dump(trace_data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=check_input_path_valid,
        dest='db_path',
        help='Path to the folder containing profile data.')
    parser.add_argument(
        '--output',
        type=check_output_path_valid,
        default=os.getcwd(),
        help='Output file path to save results.')

    args = parser.parse_args()
    return args.db_path, args.output


def check_input_path_valid(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Path is not a valid directory: {path}")
    if '..' in path or path.startswith('/'):
        raise argparse.ArgumentTypeError(f"Path contains illegal characters: {path}")
    return path


def check_output_path_valid(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.access(path, os.W_OK):
        raise argparse.ArgumentTypeError(f"Output path is not writable: {path}")
    return path


def read_cpu_data_from_db(db_path, sys_start_cnt):
    cpu_data_df = find_cpu_data_from_folder(db_path)
    cpu_data_df['start_time'] = convert_nano_to_ts(cpu_data_df['start_time'])
    cpu_data_df['end_time'] = convert_nano_to_ts(cpu_data_df['end_time'])
    return cpu_data_df


def find_cpu_data_from_folder(folder_path):
    cpu_data_df = pd.DataFrame()
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename == 'host_cpu_usage.db':
                db_path = os.path.join(root, filename)
                cpu_data_df = load_cpu_data_from_database(db_path)
    if cpu_data_df.empty:
        raise ValueError(f"No valid cpu database found in {folder_path}, please check.")
    cpu_data_df = cpu_data_df.sort_values(by='start_time', ascending=True).reset_index(drop=True)
    return cpu_data_df


def load_cpu_data_from_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
            SELECT *
            FROM CpuUsage
            WHERE cpu_no == 'Avg'
        """)

    cpu_data = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    cpu_data_df = pd.DataFrame(cpu_data, columns=columns)
    conn.close()
    return cpu_data_df


def main():
    db_path, output = parse_args()
    sys_start_cnt = get_sys_start_cnt(db_path)
    all_data_df = concat_data_from_folder(db_path)
    cpu_data_df = read_cpu_data_from_db(db_path, sys_start_cnt)
    all_data_df = data_convert(all_data_df, sys_start_cnt)
    trace_data = create_trace_events(all_data_df, cpu_data_df)
    save_trace_data_into_json(trace_data, output)


if __name__ == "__main__":
    main()
