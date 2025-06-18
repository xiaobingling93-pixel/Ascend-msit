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

import unittest
from unittest.mock import patch

from msprechecker.core.collectors.sys_collector import (
    DriverVersionCollector,
    ToolkitVersionCollector,
    MindIEVersionCollector,
    ATBVersionCollector,
    ATBSpeedVersionCollector,
    AscendInfoCollector,
)


class TestDriverVersionCollector(unittest.TestCase):
    def setUp(self):
        self.collector = DriverVersionCollector()

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_version_found(self, mock_read_file_lines):
        mock_read_file_lines.return_value = [
            "Some other line",
            "Version=1.2.3",
            "Another line"
        ]
        result = self.collector.collect()
        self.assertEqual(result, {"version": "1.2.3"})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_version_not_found(self, mock_read):
        mock_read.return_value = [
            "Build=123",
            "Date=2025-01-01"
        ]
        result = self.collector.collect()
        self.assertEqual(result, {"version": None})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_empty_file(self, mock_read):
        mock_read.return_value = []
        result = self.collector.collect()
        self.assertEqual(result, {"version": None})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_file_not_found(self, mock_read):
        mock_read.return_value = None
        result = self.collector.collect()
        self.assertEqual(result, {"version": None})


class TestToolkitVersionCollector(unittest.TestCase):
    def setUp(self):
        self.collector = ToolkitVersionCollector()

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_with_default_home(self, mock_read):
        mock_read.side_effect = [
            ["[Version]\n", "version=:1.2.3\n"],  # version.cfg content
            ["timestamp=2025-01-01"]             # version.info content
        ]
        result = self.collector.collect()
        self.assertEqual(result, {"version": "1.2.3", "time": "2025-01-01"})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_with_custom_home(self, mock_read):
        mock_read.side_effect = [
            ["[Version]\n", "version=:2.3.4\n"],
            ["timestamp=2025-02-02"]
        ]
        result = self.collector.collect()
        self.assertEqual(result, {"version": "2.3.4", "time": "2025-02-02"})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_version_only(self, mock_read):
        mock_read.side_effect = [
            ["version=[:1.2.3]"],
            []  # Empty compiler version file
        ]
        result = self.collector.collect()
        self.assertEqual(result, {"version": "1.2.3", "time": None})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_time_only(self, mock_read):
        mock_read.side_effect = [
            [],  # Empty version.cfg
            ["timestamp=2025-01-01"]
        ]
        result = self.collector.collect()
        self.assertEqual(result, {"version": None, "time": "2025-01-01"})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_no_files(self, mock_read):
        mock_read.side_effect = [[], []]
        result = self.collector.collect()
        self.assertEqual(result, {"version": None, "time": None})


class TestMindIEVersionCollector(unittest.TestCase):
    def setUp(self):
        self.collector = MindIEVersionCollector()

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_version_found(self, mock_read):
        mock_read.return_value = [
            "mindie: 1.2.3",
            "other: info"
        ]
        result = self.collector.collect()
        self.assertEqual(result, {"version": "1.2.3"})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_with_custom_home(self, mock_read):
        mock_read.return_value = ["mindie: 2.3.4"]
        result = self.collector.collect()
        self.assertEqual(result, {"version": "2.3.4"})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_version_not_found(self, mock_read):
        mock_read.return_value = ["other: info", "no version here"]
        result = self.collector.collect()
        self.assertEqual(result, {"version": None})

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_empty_file(self, mock_read):
        mock_read.return_value = []
        result = self.collector.collect()
        self.assertEqual(result, {"version": None})


class TestATBVersionCollector(unittest.TestCase):
    def setUp(self):
        self.collector = ATBVersionCollector()

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_all_fields(self, mock_read):
        mock_read.return_value = [
            "Version: 1.2.3",
            "Branch: main",
            "Commit: abc123",
            "Other: info"
        ]
        result = self.collector.collect()
        self.assertEqual(result, {
            "version": "1.2.3",
            "branch": "main",
            "commit": "abc123"
        })

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_some_fields(self, mock_read):
        mock_read.return_value = [
            "Version: 1.2.3",
            "Other: info",
            "Commit: abc123"
        ]
        result = self.collector.collect()
        self.assertEqual(result, {
            "version": "1.2.3",
            "branch": None,
            "commit": "abc123"
        })

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_with_custom_home(self, mock_read):
        mock_read.return_value = ["Version: 2.3.4"]
        result = self.collector.collect()
        self.assertEqual(result, {
            "version": "2.3.4",
            "branch": None,
            "commit": None
        })

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_empty_file(self, mock_read):
        mock_read.return_value = []
        result = self.collector.collect()
        self.assertEqual(result, {
            "version": None,
            "branch": None,
            "commit": None
        })


class TestATBSpeedVersionCollector(unittest.TestCase):
    def setUp(self):
        self.collector = ATBSpeedVersionCollector()

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_all_fields(self, mock_read):
        mock_read.return_value = [
            "Version: 1.2.3",
            "Branch: main",
            "Commit: abc123",
            "Time: 2025-01-01",
            "Other: info"
        ]
        result = self.collector.collect()
        self.assertEqual(result, {
            "version": "1.2.3",
            "branch": "main",
            "commit": "abc123",
            "time": "2025-01-01"
        })

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_some_fields(self, mock_read):
        mock_read.return_value = [
            "Version: 1.2.3",
            "Time: 2025-01-01"
        ]
        result = self.collector.collect()
        self.assertEqual(result, {
            "version": "1.2.3",
            "branch": None,
            "commit": None,
            "time": "2025-01-01"
        })

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_with_custom_home(self, mock_read):
        mock_read.return_value = ["Version: 2.3.4"]
        result = self.collector.collect()
        self.assertEqual(result, {
            "version": "2.3.4",
            "branch": None,
            "commit": None,
            "time": None
        })

    @patch('msprechecker.core.collectors.sys_collector.read_file_lines')
    def test_collect_empty_file(self, mock_read):
        mock_read.return_value = []
        result = self.collector.collect()
        self.assertEqual(result, {
            "version": None,
            "branch": None,
            "commit": None,
            "time": None
        })


class TestAscendInfoCollector(unittest.TestCase):
    def setUp(self):
        self.collector = AscendInfoCollector()

    @patch.object(DriverVersionCollector, 'collect')
    @patch.object(ToolkitVersionCollector, 'collect')
    @patch.object(MindIEVersionCollector, 'collect')
    @patch.object(ATBVersionCollector, 'collect')
    @patch.object(ATBSpeedVersionCollector, 'collect')
    def test_collect_all_components(self, mock_atb_speed, mock_atb, mock_mindie, 
                                   mock_toolkit, mock_driver):
        mock_driver.return_value = {"version": "1.2.3"}
        mock_toolkit.return_value = {"version": "2.3.4", "time": "2025-01-01"}
        mock_mindie.return_value = {"version": "3.4.5"}
        mock_atb.return_value = {"version": "4.5.6", "branch": "main", "commit": "abc123"}
        mock_atb_speed.return_value = {"version": "5.6.7", "time": "2025-02-02"}

        result = self.collector.collect()
        
        self.assertEqual(result, {
            "driver": {"version": "1.2.3"},
            "toolkit": {"version": "2.3.4", "time": "2025-01-01"},
            "mindie": {"version": "3.4.5"},
            "atb": {"version": "4.5.6", "branch": "main", "commit": "abc123"},
            "atb-models": {"version": "5.6.7", "time": "2025-02-02"}
        })

    @patch.object(DriverVersionCollector, 'collect')
    @patch.object(ToolkitVersionCollector, 'collect')
    @patch.object(MindIEVersionCollector, 'collect')
    @patch.object(ATBVersionCollector, 'collect')
    @patch.object(ATBSpeedVersionCollector, 'collect')
    def test_collect_some_missing(self, mock_atb_speed, mock_atb, mock_mindie, 
                                mock_toolkit, mock_driver):
        mock_driver.return_value = {"version": None}
        mock_toolkit.return_value = {"version": "2.3.4", "time": None}
        mock_mindie.return_value = {"version": None}
        mock_atb.return_value = {"version": None, "branch": None, "commit": None}
        mock_atb_speed.return_value = {"version": None, "time": None}

        result = self.collector.collect()
        
        self.assertEqual(result, {
            "driver": {"version": None},
            "toolkit": {"version": "2.3.4", "time": None},
            "mindie": {"version": None},
            "atb": {"version": None, "branch": None, "commit": None},
            "atb-models": {"version": None, "time": None}
        })
