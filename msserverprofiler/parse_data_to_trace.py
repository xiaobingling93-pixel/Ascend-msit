# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
SYS_TS = psutil.boot_time()


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
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename == 'msproftx.db':
                db_path = os.path.join(root, filename)
                data_df = load_data_from_database(db_path)
                full_df = pd.concat([full_df, data_df], ignore_index=True)
    if full_df.empty:
        raise ValueError(f"No valid database found in {folder_path}, please check.")
    full_df = full_df.sort_values(by='start_time', ascending=True).reset_index(drop=True)
    return full_df


def convert_syscnt_to_ts(cnt):
    return SYS_TS + (cnt / DEFAULT_FREQ) * 1000


def extract_span_info_from_message(message):
    span_id = None
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


def extract_ids_from_reslist(message):
    res_list = message.get('resList', None)
    rid = []
    token_id = []
    if res_list:
        for req in res_list:
            rid.append(req.get('rid', None))
            token_id.append(req.get('iter', None))
        if len(res_list) > 1:
            return ','.join(rid), ','.join(token_id)
        elif rid and token_id:
            return str(rid[0]), str(token_id[0])
    return res_list, res_list


def extract_rid(message):
    rid_from_message = message.get('rid', None)
    if rid_from_message is not None:
        return str(rid_from_message)

    rid_from_reslist, _ = extract_ids_from_reslist(message)
    if rid_from_reslist is not None:
        return rid_from_reslist
    else:
        return rid_from_message


def extract_batch_type(message):
    _, token_id = extract_ids_from_reslist(message)
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
    all_data_df['during_time'] = None

    for _, data in all_data_df.iterrows():
        if data['event_type'] == 'start/end':
            span_with_dur_time[data['mark_id']] = {
                'during_time': data['end_time'] - data['start_time'],
                'start_time': data['start_time']
            }

    for idx, data in all_data_df.iterrows():
        if data['span_id'] is not None:
            span_id = int(data['span_id'])
            if span_id in span_with_dur_time:
                all_data_df.loc[idx, 'during_time'] = span_with_dur_time[span_id]['during_time']
                all_data_df.loc[idx, 'start_time'] = span_with_dur_time[span_id]['start_time']
    return all_data_df


def data_convert(all_data_df):
    all_data_df['start_time'] = convert_syscnt_to_ts(all_data_df['start_time'])
    all_data_df['end_time'] = convert_syscnt_to_ts(all_data_df['end_time'])
    all_data_df[['span_id', 'message']] = (all_data_df['message'].apply
                                           (lambda x: pd.Series(extract_span_info_from_message(x))))
    all_data_df['message'] = all_data_df['message'].apply(lambda x: convert_message_to_json(x))
    all_data_df['rid'] = all_data_df['message'].apply(lambda x: extract_rid(x))
    all_data_df['batch_type'] = all_data_df['message'].apply(lambda x: extract_batch_type(x))
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
    if name == 'ReqDeQueue':
        args['queueID'] = message.get('queue', None)
        args['queueSize'] = message.get('size', None)
    if name == 'httpRes':
        args['replyTokenSize'] = message.get('replyTokenSize', None)
    return args


def create_trace_events(all_data_df):
    trace_events = []

    for idx, data in all_data_df.iterrows():
        if data['event_type'] == 'marker' and data['name'] is not None and data['span_id'] is None:
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
        if data['span_id'] is not None:
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
            if idx == 0:
                trace_events.append(
                    {
                        "name": "flow_" + data['rid'],
                        "ph": "s",
                        "id": int(data['rid']),
                        "pid": data['pid'],
                        "tid": data['name'],
                        "ts": data['start_time']
                    }
                )
            if idx > 0:
                trace_events.append(
                    {
                        "name": "flow_" + data['rid'],
                        "ph": "f",
                        "bp": "e",
                        "id": int(data['rid']),
                        "pid": data['pid'],
                        "tid": data['name'],
                        "ts": data['start_time']
                    }
                )
                trace_events.append(
                    {
                        "name": "flow_" + data['rid'],
                        "ph": "s",
                        "id": int(data['rid']),
                        "pid": data['pid'],
                        "tid": data['name'],
                        "ts": data['start_time']
                    }
                )
        if data['name'] == 'deviceKvCache':
            trace_events.append(
                {
                    "name": "KvCache Value",
                    "ph": "C",
                    "ts": data['start_time'],
                    "pid": data['pid'],
                    "tid": data['name'],
                    "cat": "Metrics",
                    "args": {
                        'KvCache Value': data['message'].get('value', None)
                    }
                }
            )

    trace_data = {"traceEvents": trace_events}
    return trace_data


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


def main():
    db_path, output = parse_args()
    all_data_df = concat_data_from_folder(db_path)
    all_data_df = data_convert(all_data_df)
    trace_data = create_trace_events(all_data_df)
    save_trace_data_into_json(trace_data, output)


if __name__ == "__main__":
    main()