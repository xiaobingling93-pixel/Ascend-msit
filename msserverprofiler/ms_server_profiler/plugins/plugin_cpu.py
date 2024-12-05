import psutil

from ms_server_profiler.parse import PluginBase


SYS_TS = psutil.boot_time()


def _convert_syscnt_to_ts(cnt, start_cnt, cpu_frequency):
    return (SYS_TS + ((cnt - start_cnt) / cpu_frequency)) * 1000 * 1000


class PluginCpu(PluginBase):
    name = "plugin_cpu"
    depends = []

    @classmethod
    def parse(cls, data):
        cpu_data_df = data.get('cpu_data_df')
        cpu_start_cnt = data.get('cpu_start_cnt')
        cpu_frequency = data.get('cpu_frequency')   
        cpu_data_df['start_time'] = _convert_syscnt_to_ts(cpu_data_df['start_time'], cpu_start_cnt, cpu_frequency)
        cpu_data_df['end_time'] = _convert_syscnt_to_ts(cpu_data_df['end_time'], cpu_start_cnt, cpu_frequency)
        data['cpu_data_df'] = cpu_data_df
        return data
