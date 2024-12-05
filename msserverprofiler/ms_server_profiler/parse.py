from abc import abstractmethod
from collections import namedtuple
from typing import List, Dict


class PluginBase:
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def depends(self) -> List[str]:
        pass

    @abstractmethod
    def parse(self, data: Dict) -> Dict:
        pass

class ExporterBase:
    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def export(self, data: Dict) -> None:
        pass


def read_origin_db(db_path: str):
    from parse_data_to_trace import concat_data_from_folder, find_cpu_data_from_folder, get_start_cnt, get_cpu_freq
    
    tx_data_df = concat_data_from_folder(db_path)
    cpu_data_df = find_cpu_data_from_folder(db_path)
    sys_start_cnt, cpu_start_cnt = get_start_cnt(db_path)
    cpu_frequency = get_cpu_freq(db_path)

    return dict(
        tx_data_df=tx_data_df, 
        cpu_data_df=cpu_data_df, 
        sys_start_cnt=sys_start_cnt,
        cpu_start_cnt=cpu_start_cnt,
        cpu_frequency=cpu_frequency
        )


def sort_plugins(plugins: List[PluginBase]) -> List[PluginBase]:
    "TODO"
    return plugins


def parse(input_path, plugins: List[PluginBase], exporters: List[ExporterBase]):
    buildin_plugins = []

    all_plugins = []
    all_plugins.extend(buildin_plugins)
    all_plugins.extend(sort_plugins(plugins))

    data = read_origin_db(input_path)

    for plugin in all_plugins:
        data = plugin.parse(data)

    for exporter in exporters:
        exporter.export(data)
