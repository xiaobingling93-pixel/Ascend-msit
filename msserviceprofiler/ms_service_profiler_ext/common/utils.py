# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import re
import pandas as pd
from ms_service_profiler.utils.log import logger


EVENT_PAIRS = [
    ('batchFrameworkProcessing', 'preprocessBatch'),
    ('preprocessBatch', 'serializeExcueteMessage'),
    ('serializeExcueteMessage', 'deserializeExecuteRequestsForInfer'),
    ('grpcWriteToSlave', 'deserializeExecuteRequestsForInfer'),
    ('deserializeExecuteRequestsForInfer', 'convertTensorBatchToBackend'),
    ('convertTensorBatchToBackend', 'getInputMetadata'),
    ('getInputMetadata', 'preprocess'),
    ('getInputMetadata', 'prepareInputs'),
    ('prepareInputs', 'operatorExecute'),
    ('preprocess', 'forward'),
    ('forward', 'sample'),
    ('operatorExecute', 'sample'),
    ('postprocess', 'generateOutput'),
    ('generateOutput', 'processPythonExecResult'),
    ('deserializeExecuteResponse', 'saveoutAndContinueBatching'),
]

SYNC_EVENT_PAIRS = [
    ('processBroadcastMessage', 'deserializeExecuteRequestsForInfer'),
    ('getInputMetadata', 'preprocess'),
    ('getInputMetadata', 'prepareInputs'),
    ('preprocess', 'forward'),
    ('forward', 'sample'),
    ('prepareInputs', 'operatorExecute'),
    ('operatorExecute', 'sample'),
    ('getInputMetadata', 'sample'),
    ('sample', 'postprocess'),
    ('postprocess', 'processPythonExecResult'),
]

FILTER_LIST = ['deserializeExecuteRequestsForInfer', 'deserializeExecuteResponse', 'convertTensorBatchToBackend', 
               'processPythonExecResult', 'forward', 'sample', 'setInferBuffer', 'grpcWriteToSlave']

NAME_LIST = ['httpReq', 'encode', 'batchFrameworkProcessing', 'preprocessBatch', 'serializeExcueteMessage', 
             'setInferBuffer', 'grpcWriteToSlave', 'deserializeExecuteRequestsForInfer', 
             'convertTensorBatchToBackend', 'getInputMetadata', 'preprocess', 'forward', 'sample', 
             'postprocess', 'receiveInfer', 'prepareInputs', 'operatorExecute', 'generateOutput', 
             'processPythonExecResult', 'deserializeExecuteResponse', 'handleTaskExecution',
             'saveoutAndContinueBatching', 'continueBatching', 'decode', 'httpRes']

CSV_COLUMNS = ['name', 'during_time', 'pid', 'tid', 'start_time', 'end_time', 'rid', 'start_datetime', 
               'end_datetime', 'batch_type', 'batch_size', 'rid_list', 'token_id_list']

RENAMED_COLUMNS = {
        'start_time': 'start_time(microsecond)',
        'end_time': 'end_time(microsecond)',
        'during_time': 'during_time(microsecond)'
    }


def confirmation_interaction(prompt):
    confirm_pattern = re.compile(r'y(?:es)?', re.IGNORECASE)
    
    try:
        user_action = input(prompt)
    except Exception:
        return False
    
    return bool(confirm_pattern.match(user_action))


def preprocess_framework_df(framework_df):
    try:
        framework_df = framework_df[framework_df['name'].isin(NAME_LIST)]
        framework_df = framework_df[CSV_COLUMNS]
        framework_df = framework_df.rename(columns=RENAMED_COLUMNS)
    except KeyError as e:
        logger.warning(f"Field '{e.args[0]}' not found in msproftx.db.")
        return None

    return framework_df


def is_valid_prefill(batch_group, framework_df):
    batch_row = batch_group.iloc[0]
    rid = batch_row['rid_list'][0]
    if rid == '0':
        return False

    target_encode = framework_df[(framework_df['rid'] == str(rid)) & 
                                 (framework_df['name'] == 'httpReq')]
    return not target_encode.empty


def get_groups(framework_df, batch_size, name):
    result_df = []
    
    groups = framework_df.groupby((framework_df['name'] == 'batchFrameworkProcessing').cumsum())
    batch_rows = framework_df[framework_df['name'] == 'batchFrameworkProcessing']
    pid = batch_rows['pid'].iloc[0]
    for _, group in groups:
        batch_group = group[(group['name'] == 'batchFrameworkProcessing') &
                                            (group['batch_type'] == name) &
                                            (group['batch_size'] == str(batch_size)) &
                                            (group['pid'] == pid)]
        if batch_group.empty:
            continue

        if name == 'Prefill' and not is_valid_prefill(batch_group, framework_df):
            continue
        result_df.append(group) 

    return result_df


def get_event_pair_df(framework_df, name):
    new_rows = []
    # 遍历每个事件对
    for current_name, next_name in EVENT_PAIRS:
        # 收集当前事件的所有实例
        current_events = framework_df[framework_df['name'] == current_name].copy()
        len_current = len(current_events)
        if len_current == 0:
            logger.debug(f'{name}: No data named {current_name}')
            continue
        next_events = framework_df[framework_df['name'] == next_name].copy()
        len_next = len(next_events)
        if len_next == 0:
            logger.debug(f'{name}: No data named {next_name}')
            continue
        # 提取当前事件的pid，tid
        for i in range(len_current):
            cur_pid = current_events.iloc[i]['pid']
            cur_tid = current_events.iloc[i]['tid']
            if (current_name, next_name) in SYNC_EVENT_PAIRS:
                next_events_filter = next_events[(next_events['pid'] == cur_pid) & (next_events['tid'] == cur_tid)]
                if len(next_events_filter) == 0:
                    continue
                current_end = current_events.iloc[i]['end_time(microsecond)']
                next_start = next_events_filter.iloc[0]['start_time(microsecond)']
                if next_start < current_end:
                    continue
            else:
                next_events_filter = next_events[next_events['pid'] == cur_pid]
                if len(next_events_filter) == 0:
                    continue
                current_end = current_events.iloc[i]['end_time(microsecond)']
                next_start = next_events_filter.iloc[0]['start_time(microsecond)']
                if next_start < current_end:
                    continue
            # 创建新行
            new_row = {
                'name': f"Between-{current_name}-{next_name}",
                'during_time(microsecond)': next_start - current_end,
                'start_time(microsecond)': current_end,
                'end_time(microsecond)': next_start,
                'pid': current_events.iloc[i]['pid'],
                'tid': current_events.iloc[i]['tid'],
                'message': None,
                'mark_id': None,
                'start_datetime': None,
                'end_datetime': None,
                'batch_type': None,
                'batch_size': None
            }
            new_rows.append(new_row)

    # 将新行转换为 DataFrame
    new_df = pd.DataFrame(new_rows)

    # 合并原数据和新行，并按时间排序
    result_df = pd.concat([framework_df, new_df], ignore_index=True)
    result_df = result_df.sort_values(by='start_time(microsecond)').reset_index(drop=True)
    return result_df
    

def postprocess_framework_df(framework_df, post_event_pairs, name):
    filter_time = 100000
    if framework_df.empty:
        logger.warning(f'{name}: df is empty')
        return framework_df
    all_time_rows = framework_df[framework_df['name'] == 'AllTime']
    start_time = all_time_rows['start_time(microsecond)']
    end_time = all_time_rows['end_time(microsecond)']
    all_time_rows.loc[:, 'start_time(microsecond)'] = end_time
    all_time_rows.loc[:, 'end_time(microsecond)'] = start_time
    # 更新 framework_df 中的对应行
    framework_df.update(all_time_rows)
    if name == 'Prefill':
        if 'continueBatching' not in framework_df['name'].values:
            post_event_pairs.append(('deserializeExecuteResponse', 'httpRes'))
            if 'httpRes' not in framework_df['name'].values:
                post_event_pairs.append(('deserializeExecuteResponse', 'AllTime'))
    new_rows = []
    if 'preprocessBatch' not in framework_df['name'].values:
        post_event_pairs.append(('batchFrameworkProcessing', 'serializeExcueteMessage'))
    if 'continueBatching' not in framework_df['name'].values and name == 'Decode':
        post_event_pairs.append(('deserializeExecuteResponse', 'AllTime'))
    if 'generateOutput' not in framework_df['name'].values:
        post_event_pairs.append(('postprocess', 'processPythonExecResult'))
    # 遍历每个事件对
    for current_name, next_name in post_event_pairs:
        # 收集当前事件的所有实例
        current_events = framework_df[framework_df['name'] == current_name].copy()
        len_current = len(current_events)
        if len_current == 0:
            logger.debug(f'{name}: No data named {current_name}')
            continue
        next_events = framework_df[framework_df['name'] == next_name].copy()
        len_next = len(next_events)
        if len_next == 0:
            logger.debug(f'{name}: No data named {next_name}')
            continue
        current_end = current_events.iloc[0]['end_time(microsecond)']
        next_start = next_events.iloc[0]['start_time(microsecond)']
        # 创建新行
        new_row = {
            'name': f"Between-{current_name}-{next_name}",
            'during_time(microsecond)': next_start - current_end,
            'start_time(microsecond)': current_end,
            'end_time(microsecond)': next_start,
            'pid': current_events.iloc[0]['pid'],
            'tid': current_events.iloc[0]['tid'],
            'message': None,
            'mark_id': None,
            'start_datetime': None,
            'end_datetime': None,
            'batch_type': None,
            'batch_size': None
        }
        new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows)
    specific_row = new_df[new_df['name'] == 'Between-batchFrameworkProcessing-serializeExcueteMessage']
    if specific_row.empty:
        return pd.DataFrame()
    if (int)(specific_row['during_time(microsecond)'].iloc[0]) > filter_time:
        logger.debug(f'{name}: time is large than filter_time, continue')
        return pd.DataFrame()
    framework_df = pd.concat([framework_df, new_df], ignore_index=True)
    framework_df = framework_df.sort_values(by=['start_time(microsecond)']).reset_index(drop=True)
    handletask_rows = framework_df[framework_df['name'] == 'handleTaskExecution']
    framework_df = framework_df[~framework_df.index.isin(handletask_rows.index)]  # 删除 'handleTaskExecution' 行
    if name == 'Decode':
        framework_df = framework_df.drop(framework_df[(framework_df['name'] == 'httpRes')].index)
    framework_df['during_time(microsecond)'] = framework_df['during_time(microsecond)'] / 1000
    
    if 'prepareInputs' in framework_df['name'].values and 'operatorExecute' in framework_df['name'].values:
        delete_list = ['preprocess', 'forward', 'Between-getInputMetadata-preprocess', 
                       'Between-preprocess-forward', 'Between-forward-sample']
    else:
        delete_list = ['prepareInputs', 'operatorExecute', 'Between-getInputMetadata-prepareInputs', 
                       'Between-prepareInputs-operatorExecute', 'Between-operatorExecute-sample']

    framework_df = framework_df[~framework_df['name'].isin(delete_list)]
    return framework_df


def get_filter_rule_df(framework_df):
    serialize_execute_index = framework_df[framework_df['name'] == 'serializeExcueteMessage'].index
    if len(serialize_execute_index) != 0:
        serialize_execute_index = serialize_execute_index[0]
        rows_to_drop = framework_df[(framework_df.index < serialize_execute_index) &
                                    (framework_df['name'].isin(FILTER_LIST))].index
        framework_df = framework_df.drop(rows_to_drop)
    
    # 删除 'name' 为 'encode'、'httpReq'、'decode'、'httpRes' 的行
    filter_list = ['encode', 'httpReq', 'DecodeEnd', 'httpRes']
    framework_df = framework_df.drop(framework_df[framework_df['name'].isin(filter_list)].index)
    return framework_df


def get_batch_framework(framework_df, name):
    if name == 'Prefill':
        framework_df = get_filter_rule_df(framework_df)
    framework_df = get_event_pair_df(framework_df, name)

    grouped_df = framework_df.groupby('name')
    sorted_groups = grouped_df.apply(lambda x: x.sort_values(by='during_time(microsecond)', ascending=True))
    if 'generateOutput' in grouped_df.groups:
        generate_output_pid = grouped_df.get_group('generateOutput')['pid'].iloc[0]
        generate_output_tid = grouped_df.get_group('generateOutput')['tid'].iloc[0]
    elif 'sample' in grouped_df.groups:
        sample_group = grouped_df.get_group('sample')
        last_sample_row = sample_group.tail(1)
        generate_output_pid = last_sample_row['pid'].values[0]
        generate_output_tid = last_sample_row['tid'].values[0]
    else:
        return pd.DataFrame()

    filter_name = ['deserializeExecuteRequestsForInfer', 'convertTensorBatchToBackend'
                   'Between-convertTensorBatchToBackend-convertTensorBatchToBackend',
                   'Between-convertTensorBatchToBackend-getInputMetadata']
    result_df = sorted_groups[(sorted_groups['pid'] == generate_output_pid) &
                              (sorted_groups['tid'] == generate_output_tid)]
    result_df1 = sorted_groups[(sorted_groups['pid'] == generate_output_pid) &
                               (sorted_groups['name'].isin(filter_name))]
    other_group = sorted_groups[sorted_groups['pid'] != generate_output_pid]
    other_group = other_group[~other_group['name'].isin(result_df['name'])]
    other_df = other_group.drop_duplicates(subset='name', keep='first')
    result_df = pd.concat([result_df, other_df])
    return result_df


def get_filter_df(framework_df, name):
    """
    从第一条httpReq开始
    """
    if name == 'Prefill':
        filter_name = 'httpReq'
    else:
        filter_name = 'batchFrameworkProcessing'
    first_batch_schedule_index = 0
    try:
        first_batch_schedule_index = framework_df[framework_df['name'] == filter_name].index[0]
    except IndexError:
        logger.warning(f"{name}: No data named {filter_name}")
    filter_df = framework_df.loc[first_batch_schedule_index:]
    return filter_df


def get_statistics_data(framework_df, filter_name, name):
    start_index = framework_df[framework_df['name'] == filter_name].index[-1]
    framework_df['max'] = framework_df.groupby('name')['during_time(microsecond)'].transform('max')
    framework_df['min'] = framework_df.groupby('name')['during_time(microsecond)'].transform('min')
    framework_df['mean'] = framework_df.groupby('name')['during_time(microsecond)'].transform('mean')
    framework_df['std'] = framework_df.groupby('name')['during_time(microsecond)'].transform('std')
    framework_df.insert(2, 'max', framework_df.pop('max'))
    framework_df.insert(3, 'min', framework_df.pop('min'))
    framework_df.insert(4, 'mean', framework_df.pop('mean'))
    framework_df.insert(5, 'std', framework_df.pop('std'))
    if name == 'Decode':
        framework_df = framework_df.iloc[:, :10]
    else:
        framework_df = framework_df.iloc[:, :11]
    return framework_df[start_index:]


def get_batch_all_time(framework_df, name):
    new_rows = []
    batch_rows = framework_df[framework_df['name'] == 'batchFrameworkProcessing']
    pid = batch_rows['pid'].iloc[0]
    current_events = framework_df[(framework_df['name'] == 'batchFrameworkProcessing') & 
    (framework_df['token_id_list'].apply(lambda x: x is not None and None not in x)) &
    (framework_df['batch_type'] == name) & (framework_df['pid'] == pid)].copy()
    len_current = len(current_events)

    if len_current < 2:
        logger.warning(f"{name}: The length of batchFrameworkProcessing is less two")
        return framework_df
    
    # 补充插入AllTime行
    for i in range(len_current - 2):
        current_row = current_events.iloc[i]
        next_row = current_events.iloc[i + 1]
        during_time = next_row['start_time(microsecond)'] - current_row['start_time(microsecond)']

        all_time_row = {
                'name': 'AllTime',
                'start_time(microsecond)': current_row['start_time(microsecond)'] + 2,
                'end_time(microsecond)': next_row['start_time(microsecond)'],
                'during_time(microsecond)': during_time,
                'start_datetime': current_row['start_datetime'],
                'end_datetime': next_row['start_datetime'],
            }

        # 将 'AllTime' 行添加到新行列表
        new_rows.append(all_time_row)

    new_df = pd.DataFrame(new_rows)
    result_df = pd.concat([framework_df, new_df], ignore_index=True)
    result_df = result_df.sort_values(by=['start_time(microsecond)', 'name']).reset_index(drop=True)

    return result_df


def get_batch_concat_df(filter_df, framework_df, cacl_num, name):
    concat_df = pd.DataFrame()
    empty_row = pd.DataFrame(index=[0])
    for i in range(cacl_num):
        rid = filter_df[i].iloc[0]['rid_list'][0]
        cur_df = get_batch_framework(filter_df[i], name)
        if cur_df.equals(pd.DataFrame()):
            continue
        if name == 'Prefill':
            cur_df = pd.concat([cur_df, framework_df[framework_df['rid'] == str(rid)]], ignore_index=True)
            post_event_pairs = [
                ('encode', 'batchFrameworkProcessing'),
                ('continueBatching', 'httpRes'),
            ]
        else:
            post_event_pairs = [
                ('continueBatching', 'AllTime'),
            ]
        cur_df = cur_df.sort_values(by='start_time(microsecond)')
        cur_df = postprocess_framework_df(cur_df, post_event_pairs, name)

        if cur_df.equals(pd.DataFrame()):
            logger.debug(f'{name}: cur_df is empty, continue')
            continue

        concat_df = pd.concat([concat_df, empty_row, cur_df], ignore_index=True)
    return concat_df


def process_exporter(framework_df, batch_size, batch_num, name):
    # 划分组
    result_df = get_groups(framework_df, batch_size, name)
    len_result_df = len(result_df)

    if len(result_df) == 0:
        logger.warning(f'{name}: no batchFrameworkProcessing with batch_size {batch_size}')
        return pd.DataFrame()
    
    cacl_num = len_result_df if len_result_df <= batch_num else batch_num
    concat_df = get_batch_concat_df(result_df, framework_df, cacl_num, name)
    return concat_df