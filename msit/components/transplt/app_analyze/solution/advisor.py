# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import os.path
import numpy as np

import pandas as pd
from app_analyze.utils.excel import read_excel, write_excel
from app_analyze.utils.log_util import logger
from app_analyze.common.kit_config import KitConfig as K

API_MAP_KEYS = [K.ACC_API, K.ASCEND_API, K.DESC, K.WORKLOAD, K.PARAMS, K.ACC_LINK, K.ASCEND_LINK, K.ASYNC_API]
REPORT_ADD_KEYS = [K.ASCEND_API, K.DESC, K.ASCEND_LIB, K.WORKLOAD, K.PARAMS, K.ACC_LINK, K.ASCEND_LINK, K.ASYNC_API]


class Advisor:

    def __init__(self, results):
        self.results = self._dedup_results(results)
        self.api_dfs = self._api_dfs(K.API_MAP)

    @staticmethod
    def _dedup_results(val_dict):
        """deduplicate scanning results caused by same include files in different source files"""
        rst_dict = {}
        df = pd.concat(list(val_dict.values()), ignore_index=True)
        if df.empty:
            return val_dict
        df.drop_duplicates(subset=[K.ACC_API, K.LOCATION], keep='first', inplace=True)
        df['file'] = df[K.LOCATION].str.split(',', expand=True)[0]

        files = df['file'].unique()
        for f in files:
            tmp_df = df[df['file'].isin([f])]
            tmp_df = tmp_df.drop(columns='file')
            tmp_df.reset_index(drop=True, inplace=True)
            rst_dict[f] = tmp_df
        return rst_dict

    @staticmethod
    def _api_map(api_path):
        """读取并整理一个API映射表"""
        df_dict = read_excel(api_path, hyperlink_cols=[K.ACC_LINK, K.ASCEND_LINK])
        # 将它们合并到一个DataFrame中
        apis = pd.concat([v for k, v in df_dict.items() if k.endswith('APIMap')], axis=0)
        drop_cols = [c for c in apis.columns if c not in API_MAP_KEYS]
        logger.debug(f'Drop columns from {api_path}:{drop_cols}')
        apis = apis.drop(drop_cols, axis=1)
        apis[K.WORKLOAD].fillna(K.DEFAULT_WORKLOAD, inplace=True)
        apis[K.ACC_API].fillna('', inplace=True)
        return apis

    @staticmethod
    def _sort(api, df):
        """从映射表中检索三方加速库API对应的条目并排序。"""
        scores = dict()
        rows = list()
        for _, row in df.iterrows():
            acc_apis = [s.strip() for s in row[K.ACC_API].split('/n')]
            if api in acc_apis:
                try:
                    scores[id(row)] = 1.0 / len(acc_apis)
                except ZeroDivisionError as ex:
                    raise ValueError("len(acc_apis) cannot be zero") from ex
                rows.append(row)
        rows.sort(key=lambda x: scores.get(id(x)))
        return rows

    @staticmethod
    def _workload_model(x):
        """工作量评估模型。"""
        # 或采用tanh模型：np.tanh(x / 15) * 15
        # 将定义域[0,30)缩放到[0,2)，对应的值域[0,0.5)
        try:
            y = (1 / (1 + np.exp(-x / 15)) - 0.5) * 2 * 15
        except ZeroDivisionError as ex:
            raise ValueError("workload_model encounters zero division error") from ex
        return np.ceil(y)

    def recommend(self):
        for _, df in self.results.items():
            if df.empty:
                continue
            # 增加表格列，使之包含APIMap的字段，并设置默认值
            for k in REPORT_ADD_KEYS:
                if k != K.WORKLOAD:
                    df[k] = ''
                else:
                    df[k] = K.DEFAULT_WORKLOAD
            # 遍历每一行，并进行修改
            for index, row in df.iterrows():
                if row[K.ACC_API] in K.EXCEPT_API:
                    continue
                # 1. 使用Series.str.contains()做字符串检索
                # 2. 自定义字符串检索
                lib_name, api_df = self.api_dfs.get(row[K.ACC_LIB], (None, None))
                if api_df is None:
                    continue
                query = self._sort(row[K.ACC_API], api_df)

                if query:
                    best = query[0]
                    for k in REPORT_ADD_KEYS:
                        row[k] = best.get(k, '')
                    row[K.ASCEND_LIB] = lib_name  # best中无该字段
                df.iloc[index] = row

            drop_cols = list()
            for c in df.columns:
                if not K.OPT_REPORT_KEY.get(c, True):
                    drop_cols.append(c)
            logger.debug(f'Drop columns from report:{drop_cols}')
            df.drop(drop_cols, axis=1, inplace=True)
        return self.results

    def workload(self):
        wl = list()
        for file_name, df in self.results.items():
            if df.empty:
                continue
            workload = df[K.WORKLOAD].sum()
            wl.append({'File': file_name, K.WORKLOAD: workload, 'Rectified': self._workload_model(workload)})
        wldf = pd.DataFrame(wl)
        if wldf.empty:
            return wldf
        total = wldf[K.WORKLOAD].sum()
        ttdf = pd.DataFrame({'File': ['Project'], K.WORKLOAD: [total], 'Rectified': self._workload_model(total)})
        wldf = pd.concat([wldf, ttdf], ignore_index=True)
        self.results['Workload'] = wldf
        return wldf

    def cuda_apis(self):
        cu_list = list()
        for file_name, df in self.results.items():
            if not df.empty and file_name != K.WORKLOAD:
                if K.CUDA_EN not in df.columns:
                    continue
                cu_list.append(df[df[K.CUDA_EN] == True])
        if not cu_list:
            return pd.DataFrame()
        cu_df = pd.concat(cu_list, ignore_index=True)
        cu_gp = cu_df.groupby(K.ACC_API).agg({K.ASCEND_API: 'first', K.LOCATION: 'size'})
        cu_gp.rename(columns={K.LOCATION: 'Count'}, inplace=True)
        cu_gp.reset_index(inplace=True)
        self.results['CUDA_APIs'] = cu_gp
        return cu_gp

    def to_excel(self):
        write_excel(self.results)

    def _api_dfs(self, api_map):
        """读取并整理多个API映射表"""
        api_dfs = dict()
        for k, v in api_map.items():
            lib_name = os.path.basename(v).split('_')[0]
            api_dfs[k] = [lib_name, self._api_map(v)]
        return api_dfs
