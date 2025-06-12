# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import unittest
from unittest.mock import patch, Mock, mock_open


class TestDistributeCollector(unittest.TestCase):
    def setUp(self):
        self.torch_mock = Mock()
        self.psutil_mock = Mock()
        self.import_patch = patch.dict("sys.modules", {"torch": self.torch_mock, "psutil": self.psutil_mock})
        self.import_patch.start()
        
        from msprechecker.prechecker.cluster_collector import distribute_collector, init_global_distribute_env
        self.distribute_collector = distribute_collector
        self.init_global_distribute_env = init_global_distribute_env

        self.get_local_to_master_ip_patch = patch(
            "msprechecker.prechecker.cluster_collector.get_local_to_master_ip",
            return_value="1.2.3.4"
        )
        self.get_interface_by_ip_patch = patch(
            "msprechecker.prechecker.cluster_collector.get_interface_by_ip",
            return_value=("e123dasdp", "1.2.3.4")
        )
        self.get_local_to_master_ip_patch.start()
        self.get_interface_by_ip_patch.start()

    def tearDown(self):
        self.import_patch.stop()
        self.get_local_to_master_ip_patch.stop()
        self.get_interface_by_ip_patch.stop()

    def test_rank_table_should_do_socket(self):
        rank_table_path = "rank_table.json"
        rank_table_content = {
            "version": "1.0",
            "server_count": "2",
            "server_list": [
                {
                    "server_id": "1.2.3.4",
                    "container_ip": "1.2.3.4",
                    "device": [
                        {
                            "device_id": "0",
                            "device_ip": "2.3.4.5",
                            "rank_id": "0"
                        },
                        {
                            "device_id": "1",
                            "device_ip": "3.4.5.6",
                            "rank_id": "1"
                        }
                    ],
                    "host_nic_ip": "reserve"
                },
                {
                    "server_id": "11.22.33.44",
                    "container_ip": "11.22.33.44",
                    "device": [
                        {
                            "device_id": "0",
                            "device_ip": "22.33.44.55",
                            "rank_id": "3"
                        },
                        {
                            "device_id": "1",
                            "device_ip": "33.44.55.66",
                            "rank_id": "4"
                        }
                    ],
                    "host_nic_ip": "reserve"
                }
            ],
            "status": "completed"
        }

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(rank_table_content))):
                self.assertIsNone(self.init_global_distribute_env(rank_table_path, None, "1.2.3.4"))
                self.assertIsNone(self.distribute_collector("123"))
