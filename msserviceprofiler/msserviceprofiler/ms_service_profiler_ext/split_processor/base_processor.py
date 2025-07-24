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

from abc import abstractmethod
import pandas as pd
from ms_service_profiler.exporters.utils import save_dataframe_to_csv

from ..common.constants import MAX_BATCH_NUMBER
from ..common.utils import logger
from ..common.split_utils import CSV_COLUMNS, RENAMED_COLUMNS, get_statistics_data




class BaseFrameworkProcessor:
    batch_start_name = "batch_start"
    batch_end_name = "batch_end"
    http_start_name = "http_start"
    http_end_name = "http_end"
    key_name = "key"
    all_time_name = "all_time"
    name_list = [batch_start_name, http_start_name, batch_end_name]
    http_list = [http_start_name, http_end_name]
    # 记录不计算中间时间的事件(异步, 时间掩盖)或无需计算
    filter_list = [http_end_name, all_time_name]

    @classmethod
    @abstractmethod
    def initialize(cls, args):
        cls.args = args

    @classmethod
    def run_split(cls, framework_df, name):
        framework_df = cls.preprocess_framework_df(framework_df)
        if framework_df.empty:
            return
        filter_df = cls.get_filter_df(framework_df, name)
        add_all_time_df = cls.get_batch_all_time(filter_df, name)
        framework_df = cls.process_exporter(add_all_time_df, name)
        lower_name = name.lower()
        if cls.args.log_level == "debug":
            save_dataframe_to_csv(add_all_time_df, cls.args.output_path, f"{lower_name}_detail.csv")
            save_dataframe_to_csv(framework_df, cls.args.output_path, f"{lower_name}_{cls.args.batch_num}.csv")
        filter_name = cls.http_start_name if name == "Prefill" else cls.batch_start_name
        framework_df = get_statistics_data(framework_df, filter_name, name)
        if not framework_df.empty:
            save_dataframe_to_csv(framework_df, cls.args.output_path, f"{lower_name}.csv")

    @classmethod
    def preprocess_framework_df(cls, framework_df):
        try:
            framework_df = framework_df[framework_df["name"].isin(cls.name_list)]
            framework_df = framework_df[CSV_COLUMNS]
        except KeyError as e:
            logger.warning(f"Field '{e.args[0]}' not found in datasource.")
            return pd.DataFrame()

        framework_df = framework_df.rename(columns=RENAMED_COLUMNS)
        
        return framework_df
    
    @classmethod
    def get_filter_df(cls, framework_df, name):
        """
        动态启停场景下 过滤不完整的batch
        """
        filter_name = cls.http_start_name if name == "Prefill" else cls.batch_start_name

        valid_indices = framework_df["name"] == filter_name
        if not valid_indices.any():
            logger.warning(f"{name}: No data named {filter_name}")
            return framework_df
        
        first_index = framework_df[valid_indices].index[0]

        # 过滤 DataFrame
        return framework_df.loc[first_index:]
    
    @classmethod
    def get_batch_all_time(cls, framework_df, name):
        batch_rows = framework_df[framework_df["name"] == cls.batch_start_name]
        
        if len(batch_rows) < 2:
            logger.warning(f"{name}: The length of {cls.batch_start_name} is less two")
            return framework_df
        
        def create_all_time_rows(group):
            new_rows = []
            for i in range(len(group) - 1):
                current_row = group.iloc[i]
                next_row = group.iloc[i + 1]
                during_time = next_row["start_time(ms)"] - current_row["start_time(ms)"]

                all_time_row = {
                    "name": cls.all_time_name,
                    "start_time(ms)": current_row["start_time(ms)"],
                    "end_time(ms)": next_row["start_time(ms)"],
                    "during_time(ms)": during_time,
                }
                new_rows.append(all_time_row)

            return pd.DataFrame(new_rows)

        all_time_dfs = batch_rows.groupby("pid").apply(create_all_time_rows).reset_index(drop=True)
        # 合并all_time行
        result_df = pd.concat([framework_df, all_time_dfs], ignore_index=True)
        result_df = result_df.sort_values(by=["start_time(ms)", "name"], ascending=[True, False]).reset_index(drop=True)

        return result_df

    @classmethod
    def process_exporter(cls, framework_df, name):
        # 划分组
        result_df = cls._get_groups(framework_df, name)
        len_result_df = len(result_df)

        if len_result_df == 0:
            if cls.args.batch_size > 0:
                size_recommend = cls._get_batch_size_recommend(framework_df, name)
                logger.warning("%s: no %s with batch_size %d" % (name, cls.batch_start_name, cls.args.batch_size))
                if size_recommend[0] == -1:
                    logger.warning("no %s data, please check." % name)
                else:
                    logger.warning("%s: recommend batch_size from data %s" % (name, 
                                    ', '.join(map(str, size_recommend))))
            elif cls.args.rid != "-1":
                logger.warning("%s: no %s with rid %r" % (name, cls.batch_start_name, cls.args.rid))
            return pd.DataFrame()
        
        calc_num = min(len_result_df, cls.args.batch_num, MAX_BATCH_NUMBER)
        concat_df = cls._get_concat_df(result_df, framework_df, calc_num, name)
        return concat_df
    
    @classmethod
    def _get_groups(cls, framework_df, name):
        result_df = []
        rid = cls.args.rid
        batch_size = cls.args.batch_size

        def filter_by_rid(sub_group):
            filtered = sub_group[(sub_group["name"] == cls.batch_start_name) &
                                 (sub_group["batch_type"] == name)]
            # 转换 rid_list 元素为字符串
            filtered.loc[:, "rid_list"] = filtered["rid_list"].apply(lambda x: [str(i) for i in x])
            return filtered[filtered["rid_list"].apply(lambda x: rid in x)]

        def filter_by_batch_size(sub_group):
            return sub_group[(sub_group["name"] == cls.batch_start_name) &
                             (sub_group["batch_type"] == name) &
                             (sub_group["batch_size"] == str(batch_size))]
        
        groups = framework_df.groupby((framework_df["name"] == cls.batch_start_name).cumsum())
        for _, group in groups:
            if rid != "-1":
                batch_group = filter_by_rid(group)
            elif batch_size > 0:
                batch_group = filter_by_batch_size(group)
            else:
                continue

            if batch_group.empty:
                continue

            if name == "Prefill" and not cls._is_valid_prefill(batch_group, framework_df):
                continue
            result = cls._get_full_batch(group, framework_df)

            if not result.empty:
                result_df.append(result) 

        return result_df

    @classmethod
    def _is_valid_prefill(cls, batch_group, framework_df):
        # batch_group不会为空且包含rid_list列
        batch_row = batch_group.iloc[0]
        cur_rid = batch_row["rid_list"][0]
        if cls.args.rid != "-1":
            cur_rid = cls.args.rid

        target_encode = framework_df[(framework_df["rid"] == str(cur_rid)) & 
                                     (framework_df["name"] == cls.http_start_name)]
        return not target_encode.empty

    @classmethod
    def _get_full_batch(cls, group, framework_df):
        start_index = cls.name_list.index(cls.batch_start_name)
        key_index = cls.name_list.index(cls.key_name)
        end_index = cls.name_list.index(cls.batch_end_name)
        all_time_rows = group[group["name"] == cls.all_time_name]
        if all_time_rows.empty:
            logger.debug(f"No row named {cls.all_time_name} found in the group, skip this batch")
            return pd.DataFrame()
        all_time_index = all_time_rows.index[0]
        # group里一定存在cls.batch_start_name
        batch_start_index = group[group["name"] == cls.batch_start_name].index[0]
        
        concat_list = [all_time_index]
        index = batch_start_index
        full_batch = cls.name_list[start_index: end_index + 1]

        # 找到key_row
        for name in cls.name_list[start_index: key_index]:
            mask = (framework_df.index >= index) & (framework_df["name"] == name)
            if not mask.any():
                continue
            index = framework_df[mask].index[0]

        key_pid, key_tid, key_index = cls._get_key_info(framework_df, index)
        if key_pid is None:
            logger.warning(f"no named {cls.key_name} line, skip this batch")
            return pd.DataFrame()

        # 获取完整的batch
        result_index = cls._get_batch_index(full_batch, batch_start_index, framework_df, key_pid, key_tid)
        if result_index.empty:
            return pd.DataFrame()
        framework_df.loc[framework_df["name"] == cls.all_time_name, ["start_time(ms)", "end_time(ms)"]] = \
            framework_df.loc[framework_df["name"] == cls.all_time_name, ["end_time(ms)", "start_time(ms)"]].values
        result = pd.concat([framework_df.loc[concat_list], result_index])
        result = result.sort_values(by=["start_time(ms)", "name"], ascending=[True, False]).reset_index(drop=True)
        return result
    
    @classmethod
    def _get_key_info(cls, framework_df, index):
        key_mask = (framework_df.index > index) & (framework_df["name"] == cls.key_name)
        if not key_mask.any():
            logger.warning(f"no {cls.key_name} line, skip this batch")
            return None, None, None
        key_row = framework_df[key_mask].iloc[0]
        key_pid, key_tid = key_row["pid"], key_row["tid"]
        key_index = framework_df[key_mask].index[0]
        return key_pid, key_tid, key_index
    
    @classmethod
    def _get_batch_index(cls, full_batch, start_index, framework_df, key_pid, key_tid):
        # 从总表中获取完整的一个batch
        df_list = []
        current_index = start_index
        index_mask = framework_df.index >= start_index
        for name in full_batch:
            name_mask = framework_df["name"] == name
            pid_mask = framework_df["pid"] == key_pid
            tid_mask = framework_df["tid"] == key_tid

            conditions = [
                index_mask & name_mask & pid_mask & tid_mask,
                index_mask & name_mask & pid_mask,
                index_mask & name_mask
            ]
            index = None
            for condition in conditions:
                mask = condition
                if mask.any():
                    index = framework_df[mask].index[0]
                    break
            # 找不到的点跳过
            if index is None:
                continue
            
            df_list.append(index)
            current_index = index
            # 更新index_mask
            index_mask = framework_df.index > current_index
            
        if not df_list:
            return pd.DataFrame()

        return framework_df.loc[df_list]

    @classmethod
    def _get_concat_df(cls, filter_dfs, framework_df, calc_num, name):
        concat_df = pd.DataFrame()
        empty_row = pd.DataFrame(index=[0])
        for i in range(calc_num):
            # 1. 确认需要计算中间时间的事件, filter 名单中的事件跳过 (异步，时间掩盖)
            filter_df = filter_dfs[i]
            cur_rid = filter_df.iloc[0]["rid_list"][0]
            if cls.args.rid != "-1":
                cur_rid = cls.args.rid
            if name == "Prefill":
                http_df = framework_df[(framework_df["rid"] == str(cur_rid)) & 
                                       (framework_df["name"].isin(cls.http_list))]
                filter_df = pd.concat([filter_df, http_df], ignore_index=True)
                filter_df = filter_df.drop_duplicates(subset="name")
                filter_df = filter_df.sort_values(by="start_time(ms)")
            filter_df_ = filter_df[~filter_df["name"].isin(cls.filter_list)]
            # 2. 当前行与下一行计算during_time
            add_df = cls._calc_during_time(filter_df_)
            cur_df = pd.concat([filter_df, add_df], ignore_index=True)
            
            # 3. 与AllTime行的计算逻辑
            cur_df = cls._postprocess_framework_df(cur_df, name)

            concat_df = pd.concat([concat_df, empty_row, cur_df], ignore_index=True)
        return concat_df
    
    @classmethod
    def _calc_during_time(cls, filter_df):
        new_rows = []
        # 遍历每个事件对
        for i in range(len(filter_df) - 1):
            current_row = filter_df.iloc[i]
            next_row = filter_df.iloc[i + 1]

            current_name = current_row["name"]
            next_name = next_row["name"]

            current_end = current_row["end_time(ms)"]
            next_start = next_row["start_time(ms)"]

            # 创建新行
            new_row = {
                "name": f"Between-{current_name}-{next_name}",
                "during_time(ms)": next_start - current_end,
                "start_time(ms)": current_end,
                "end_time(ms)": next_start,
                "pid": current_row["pid"],
                "tid": current_row["tid"],
            }
            new_rows.append(new_row)

        new_df = pd.DataFrame(new_rows)

        return new_df
    
    @classmethod
    def _postprocess_framework_df(cls, framework_df, name):
        if name == "Prefill":
            post_event = pd.concat([
                framework_df[framework_df["name"] == cls.http_end_name],
                framework_df[framework_df["name"] == cls.all_time_name]
            ])
        else:
            filter_df = framework_df[(framework_df["name"] != cls.all_time_name) &
                                     (~framework_df["name"].str.startswith("Between-"))]
            if filter_df.empty:
                return pd.DataFrame()
            
            last_row = filter_df.iloc[[-1]]
            post_event = pd.concat([
                last_row,
                framework_df[framework_df["name"] == cls.all_time_name]
            ])
        framework_df = framework_df.sort_values(by="start_time(ms)")
        new_rows = cls._calc_during_time(post_event)

        framework_df = pd.concat([framework_df, new_rows], ignore_index=True)
        all_time_row = framework_df[framework_df["name"] == cls.all_time_name]
        non_all_time_row = framework_df[framework_df["name"] != cls.all_time_name]
        framework_df = pd.concat([non_all_time_row, all_time_row], ignore_index=True)
        
        return framework_df

    @classmethod   
    def _get_batch_size_recommend(cls, framework_df, name):
        batch_df = framework_df[(framework_df["name"] == cls.batch_start_name) &
                                (framework_df["batch_type"] == name)]
        if batch_df.empty:
            return [-1]
        batch_size = batch_df["batch_size"].unique()
        if len(batch_size) == 0:
            logger.warning(f"{name}: The batch_size is empty")
            return [-1]
        return batch_size