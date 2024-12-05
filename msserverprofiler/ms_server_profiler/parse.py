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


from typing import List
from collections import defaultdict, deque

class DependencyNotFoundError(Exception):
    def __init__(self, plugin_name, missing_dependency):
        self.plugin_name = plugin_name
        self.missing_dependency = missing_dependency
        super().__init__(f"Dependency '{missing_dependency}' not found for plugin '{plugin_name}'")

def sort_plugins(plugins: List[PluginBase]) -> List[PluginBase]:
    # Build the dependency graph
    graph = defaultdict(list)
    indegree = {plugin.name: 0 for plugin in plugins}

    for plugin in plugins:
        for dependency in plugin.depends:
            if dependency not in indegree:
                raise DependencyNotFoundError(plugin.name, dependency)
            graph[dependency].append(plugin.name)
            indegree[plugin.name] += 1

    # Perform topological sorting
    queue = deque([plugin.name for plugin in plugins if indegree[plugin.name] == 0])
    sorted_plugins = []

    while queue:
        current = queue.popleft()
        sorted_plugins.append(current)

        for neighbor in graph[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # Check if topological sorting was successful (i.e., no cycles)
    if len(sorted_plugins)!= len(indegree):
        raise ValueError("A cycle was detected in the plugin dependencies.")

    # Create a mapping to return the sorted plugins
    sorted_plugin_objects = {plugin.name: plugin for plugin in plugins}
    return [sorted_plugin_objects[name] for name in sorted_plugins]



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
