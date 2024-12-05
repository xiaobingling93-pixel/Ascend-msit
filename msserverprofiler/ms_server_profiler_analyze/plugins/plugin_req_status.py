from ms_server_profiler.parse import PluginBase
from ms_server_profiler.plugins.plugin_common import PluginCommon


class PluginReqStatus(PluginBase):
    name = "plugin_req_status"
    depends = [PluginCommon]

    @classmethod
    def parse(cls, data):
        return data


