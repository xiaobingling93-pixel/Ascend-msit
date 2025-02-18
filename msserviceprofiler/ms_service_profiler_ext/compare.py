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

import os
import argparse
from pathlib import Path
import multiprocessing
import sqlite3
import shutil

import pandas as pd

from ms_service_profiler.exporters.utils import check_input_path_valid, check_output_path_valid
from ms_service_profiler.utils.log import set_log_level


db_write_lock = multiprocessing.Lock()


def add_compare_visual_db_table(db_filepath, df, table_name):
    with db_write_lock:
        try:
            conn = sqlite3.connect(db_filepath)
            conn.isolation_level = None
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.commit()
            conn.close()
        except Exception as ex:
            raise ValueError(f"Cannot write {table_name} into {db_filepath}.") from ex


def compare_abs_and_persent(row):
    a, b = row
    if b == 0:
        return f'{a - b:+.2f} | '
    return f'{a - b:+.2f} | {(a - b) / b * 100:+.2f}%'


def compare_csv(fp_a, fp_b):
    df_a = pd.read_csv(fp_a)
    df_b = pd.read_csv(fp_b)
    
    # 确保列名一致
    if set(df_a.columns) != set(df_b.columns):
        raise ValueError("两个 CSV 文件的列名不一致！")

    # 按 Metric 列合并两个 DataFrame
    df_merged = pd.merge(df_a, df_b, on='Metric', suffixes=('_a', '_b'))

    # 动态计算差值
    for col in df_a.columns[1:]:  # 跳过 Metric 列
        # 计算绝对差值
        abs_diff = (df_merged[f'{col}_a'] - df_merged[f'{col}_b'])
        
        # 计算百分比相对差值（避免除以零错误）
        rel_diff = abs_diff / df_merged[f'{col}_a'].replace(0, pd.NA) * 100
        with pd.option_context('future.no_silent_downcasting', True):
            rel_diff = rel_diff.fillna(0)  # 将 NaN 替换为 0
        
        # 合并为字符串格式
        df_merged[f'{col}_diff'] = abs_diff.round(2).astype(str) + '|' + rel_diff.round(2).astype(str) + '%'

    # 存储所有行的列表
    rows = []

    # 遍历合并后的 DataFrame
    for _, row in df_merged.iterrows():
        metric = row['Metric']
        
        # 添加 a 数据
        a_row = {'Metric': metric, 'Data Source': str(fp_a.parent)}
        for col in df_a.columns[1:]:
            a_row[col] = row[f'{col}_a']
        rows.append(a_row)
        
        # 添加 b 数据
        b_row = {'Metric': metric, 'Data Source': str(fp_b.parent)}
        for col in df_a.columns[1:]:
            b_row[col] = row[f'{col}_b']
        rows.append(b_row)
        
        # 添加 diff 数据
        diff_row = {'Metric': metric, 'Data Source': 'Different'}
        for col in df_a.columns[1:]:
            diff_row[col] = row[f'{col}_diff']
        rows.append(diff_row)

    # 将所有行转换为 DataFrame
    result = pd.DataFrame(rows)

    return result


def compare(input_a, input_b):
    compare_items = {
        'service': 'service_summary.csv',
        'batch': 'batch_summary.csv',
        'request': 'request_summary.csv',
    }
    res = {}
    for name, filename in compare_items.items():
        fp_a = Path(input_a) / filename
        fp_b = Path(input_b) / filename
        if filename.endswith('.csv'):
            df = compare_csv(fp_a, fp_b)
            res[f"{name}"] = df
    return res


def report(results, output_path):
    # 将每个 DataFrame 写入不同的工作表
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for name, df in results.items():
            df.to_excel(writer, sheet_name=name, index=False)


def visualize(results, output_path):
    for name, df in results.items():
        if name == "service":
            for i in range(0, len(df), 3):
                if i + 3 <= len(df):
                    new_col_name = df.loc[i, "Metric"]
                    df[new_col_name] = df.iloc[i:i+3, :]['value'].reset_index(drop=True)
            df.drop(columns=['Metric', 'value'], inplace=True)
            df = df.iloc[:3, :]
        add_compare_visual_db_table(output_path, df, name)
        shutil.copy(
            'ms_service_profiler_ext/compare_tools/compare_visualization.json',
            output_path.with_name("compare_visualization.json")
        )


def main():
    parser = argparse.ArgumentParser(description='MS Server Profiler Analyze')
    parser.add_argument(
        'input_a',
        type=check_input_path_valid,
        help='Path to the folder containing profile data.')
    parser.add_argument(
        'input_b',
        type=check_input_path_valid,
        help='Path to the folder containing profile data.')
    parser.add_argument(
        '--output-path',
        type=check_output_path_valid,
        default=os.path.join(os.getcwd(), 'compare_result'),
        help='Output file path to save results.')
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'fatal', 'critical'],
        help='Log level to print.')

    args = parser.parse_args()

    # 初始化日志等级
    set_log_level(args.log_level)

    results = compare(args.input_a, args.input_b)
    report(results, Path(args.output_path) / 'compare_result.xlsx')
    visualize(results, Path(args.output_path) / 'compare_result.db')


if __name__ == '__main__':
    main()
