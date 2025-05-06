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
from ms_service_profiler.exporters.base import ExporterBase
from ms_service_profiler.exporters.utils import save_dataframe_to_csv
from ms_service_profiler.utils.log import logger

from ..common.split_utils import get_batch_all_time, process_exporter, get_filter_df
from ..common.split_utils import get_statistics_data, preprocess_framework_df


class ExporterDecode(ExporterBase):
    name = "decode_data"
 
    @classmethod
    def initialize(cls, args):
        cls.args = args
 
    @classmethod
    def export(cls, data) -> None:
        output = cls.args.output_path
        log_level = cls.args.log_level
        batch_size = cls.args.decode_batch_size
        batch_num = cls.args.decode_number
        rid = cls.args.decode_rid
        df = data.get('tx_data_df')
        if df is None:
            logger.error("The data is empty, please check")
            return
        framework_df = preprocess_framework_df(df)
        if framework_df is None:
            return
        filter_df = get_filter_df(framework_df, 'Decode')
        add_all_time_df = get_batch_all_time(filter_df, 'Decode')
        framework_df = process_exporter(add_all_time_df, batch_size, batch_num, rid, 'Decode')
        if log_level == 'debug':
            save_dataframe_to_csv(add_all_time_df, output, "decode1.csv")
            save_dataframe_to_csv(framework_df, output, f"decode_{batch_num}.csv")
        framework_df = get_statistics_data(framework_df, 'batchFrameworkProcessing', 'Decode')
        if not framework_df.empty:
            save_dataframe_to_csv(framework_df, output, "decode.csv")
        logger.info("Export decode data successfully.")
