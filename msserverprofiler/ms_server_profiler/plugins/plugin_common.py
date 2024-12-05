from ms_server_profiler.parse import PluginBase


class PluginCommon(PluginBase):
    name = "plugin_common"
    depends = []

    @classmethod
    def parse(cls, data):
        all_data_df = data["tx_data_df"]
        sys_start_cnt = data["sys_start_cnt"]
        cpu_frequency = data["cpu_frequency"]

        from parse_data_to_trace import data_convert
        all_data_df = data_convert(all_data_df, sys_start_cnt, cpu_frequency)
        data["tx_data_df"] = all_data_df
        return data
