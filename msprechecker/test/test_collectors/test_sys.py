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

import unittest
from unittest.mock import patch, mock_open, MagicMock
import subprocess
from msprechecker.collectors.sys import (
    LscpuCollector,
    VirtualMachineCollector,
    CPUHighPerformanceCollector,
    KernelInfoCollector,
    MemoryInfoCollector,
    SysCollector
)


class TestLscpuCollector(unittest.TestCase):
    def setUp(self):
        self.collector = LscpuCollector()

    @patch('subprocess.check_output')
    def test_parse_output_with_primary_key_should_collect_successfully(self, mock_check_output):
        test_output = "Model name: I Love U"
        mock_check_output.return_value = test_output
        result = self.collector.collect()
        self.assertEqual(result.data['model_name'], "I Love U")

    @patch('subprocess.check_output')
    def test_parse_output_with_secondary_key_should_collect_successfully(self, mock_check_output):
        test_output = "BIOS Model name: I Love U"
        mock_check_output.return_value = test_output
        result = self.collector.collect()
        self.assertEqual(result.data['model_name'], "I Love U")

    @patch('subprocess.check_output')
    def test_parse_output_with_multiple_keys_should_collect_primary(self, mock_check_output):
        test_output = "Model name: I Love U\nBIOS Model name: He doesn't love me"
        mock_check_output.return_value = test_output
        result = self.collector.collect()
        self.assertEqual(result.data['model_name'], "I Love U")

    @patch('subprocess.check_output')
    def test_parse_output_with_chinese_key_should_collect_successfully(self, mock_check_output):
        test_output = "型号名称: I Love U"
        mock_check_output.return_value = test_output
        result = self.collector.collect()
        self.assertEqual(result.data['model_name'], "I Love U")

    @patch('subprocess.check_output')
    def test_when_command_fails_should_return_empty(self, mock_check_output):
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'lscpu')
        result = self.collector.collect()
        self.assertEqual(result.data, {})
        self.assertEqual(len(result.error_handler.errors), 1)

# class TestVirtualMachineCollector(unittest.TestCase):
#     def setUp(self):
#         self.collector = VirtualMachineCollector()

#     @patch('builtins.open', new_callable=mock_open, read_data="hypervisor")
#     def test_detect_vm_positive(self, mock_file):
#         result = self.collector.collect()
#         self.assertEqual(result.data, {"virtual_machine": True})

#     @patch('builtins.open', new_callable=mock_open, read_data="processor")
#     def test_detect_vm_negative(self, mock_file):
#         result = self.collector.collect()
#         self.assertEqual(result.data, {"virtual_machine": False})

#     @patch('builtins.open')
#     def test_when_file_open_fails_should_add_error(self, mock_file):
#         mock_file.side_effect = IOError("Permission denied")
#         result = self.collector.collect()
#         self.assertEqual(result.data, {})
#         self.assertEqual(len(result.error_handler.errors), 1)


# class TestCPUHighPerformanceCollector(unittest.TestCase):
#     def setUp(self):
#         self.collector = CPUHighPerformanceCollector()

#     @patch('psutil.cpu_freq')
#     def test_psutil_strategy_positive(self, mock_cpu_freq):
#         mock_cpu_freq.return_value = MagicMock(current=3000, max=3000)
#         result = self.collector.collect()
#         self.assertEqual(result.data, {"high_performance": True})

#     @patch('psutil.cpu_freq')
#     def test_psutil_strategy_negative(self, mock_cpu_freq):
#         mock_cpu_freq.return_value = MagicMock(current=2000, max=3000)
#         result = self.collector.collect()
#         self.assertEqual(result.data, {"high_performance": False})

#     @patch('subprocess.check_output')
#     def test_dmidecode_strategy_positive(self, mock_check_output):
#         mock_check_output.return_value = """
# Processor Information
#     Max Speed: 3000 MHz
#     Current Speed: 3000 MHz
# """
#         # Disable other strategies
#         with patch.object(PsutilStrategy, 'check', return_value=False), \
#              patch.object(ScalingGovernorStrategy, 'check', return_value=False):
#             result = self.collector.collect()
#         self.assertEqual(result.data, {"high_performance": True})

#     @patch('subprocess.check_output')
#     def test_cpupower_strategy_positive(self, mock_check_output):
#         mock_check_output.return_value = """
# hardware limits: 1.20 GHz - 3.00 GHz
# current CPU frequency: 3.00 GHz
# """
#         # Disable other strategies
#         with patch.object(PsutilStrategy, 'check', return_value=False), \
#              patch.object(DmidecodeStrategy, 'check', return_value=False):
#             result = self.collector.collect()
#         self.assertEqual(result.data, {"high_performance": True})

#     @patch('os.cpu_count')
#     @patch('builtins.open', new_callable=mock_open, read_data="performance")
#     def test_scaling_governor_strategy_positive(self, mock_open, mock_cpu_count):
#         mock_cpu_count.return_value = 4
#         # Disable other strategies
#         with patch.object(PsutilStrategy, 'check', return_value=False), \
#              patch.object(DmidecodeStrategy, 'check', return_value=False):
#             result = self.collector.collect()
#         self.assertEqual(result.data, {"high_performance": True})


# class TestKernelInfoCollector(unittest.TestCase):
#     def setUp(self):
#         self.collector = KernelInfoCollector()

#     @patch('platform.uname')
#     @patch('builtins.open', new_callable=mock_open, read_data="[always] madvise never")
#     def test_collect_kernel_info(self, mock_file, mock_uname):
#         mock_uname.return_value = MagicMock(
#             system="Linux",
#             node="test-node",
#             release="5.4.0",
#             version="#1 SMP",
#             machine="x86_64",
#             processor="x86_64"
#         )
#         result = self.collector.collect()
#         expected_keys = ['system', 'node', 'release', 'version', 'machine', 'processor', 'transparent_hugepage']
#         self.assertCountEqual(result.data.keys(), expected_keys)
#         self.assertEqual(result.data['transparent_hugepage'], "always")

#     @patch('platform.uname')
#     @patch('builtins.open')
#     def test_when_hugepage_file_fails(self, mock_file, mock_uname):
#         mock_file.side_effect = IOError("Permission denied")
#         mock_uname.return_value = MagicMock(
#             system="Linux",
#             node="test-node",
#             release="5.4.0",
#             version="#1 SMP",
#             machine="x86_64",
#             processor="x86_64"
#         )
#         result = self.collector.collect()
#         self.assertNotIn('transparent_hugepage', result.data)
#         self.assertEqual(len(result.error_handler.errors), 1)


# class TestMemoryInfoCollector(unittest.TestCase):
#     def setUp(self):
#         self.collector = MemoryInfoCollector()

#     @patch('os.sysconf')
#     @patch('builtins.open', new_callable=mock_open, read_data="1")
#     def test_collect_memory_info(self, mock_file, mock_sysconf):
#         mock_sysconf.return_value = 4096
#         result = self.collector.collect()
#         self.assertEqual(result.data['page_size'], 4096)
#         self.assertEqual(result.data['overcommit_memory'], "1")

#     @patch('os.sysconf')
#     @patch('builtins.open')
#     def test_when_sysconf_fails(self, mock_file, mock_sysconf):
#         mock_sysconf.side_effect = ValueError
#         mock_file.return_value.read.return_value = "1"
#         result = self.collector.collect()
#         self.assertNotIn('page_size', result.data)
#         self.assertEqual(len(result.error_handler.errors), 1)

#     @patch('os.sysconf')
#     @patch('builtins.open')
#     def test_when_overcommit_file_fails(self, mock_file, mock_sysconf):
#         mock_sysconf.return_value = 4096
#         mock_file.side_effect = IOError("Permission denied")
#         result = self.collector.collect()
#         self.assertNotIn('overcommit_memory', result.data)
#         self.assertEqual(len(result.error_handler.errors), 1)


# class TestSysCollector(unittest.TestCase):
#     def setUp(self):
#         self.collector = SysCollector()

#     @patch('msprechecker.collectors.sys.LscpuCollector.collect')
#     @patch('msprechecker.collectors.sys.VirtualMachineCollector.collect')
#     @patch('msprechecker.collectors.sys.CPUHighPerformanceCollector.collect')
#     @patch('msprechecker.collectors.sys.KernelInfoCollector.collect')
#     @patch('msprechecker.collectors.sys.MemoryInfoCollector.collect')
#     def test_collect_all_info(self, mock_mem, mock_kernel, mock_cpu, mock_vm, mock_lscpu):
#         mock_lscpu.return_value.data = {"model_name": "Test CPU"}
#         mock_vm.return_value.data = {"virtual_machine": False}
#         mock_cpu.return_value.data = {"high_performance": True}
#         mock_kernel.return_value.data = {"system": "Linux"}
#         mock_mem.return_value.data = {"page_size": 4096}

#         result = self.collector.collect()
#         expected_keys = ['model_name', 'virtual_machine', 'high_performance', 'system', 'page_size']
#         self.assertCountEqual(result.data.keys(), expected_keys)

#     @patch('concurrent.futures.ThreadPoolExecutor')
#     def test_when_sub_collector_fails_should_continue(self, mock_executor):
#         # Setup mock futures
#         future1 = MagicMock()
#         future1.result.return_value.data = {"key1": "value1"}
#         future2 = MagicMock()
#         future2.result.side_effect = Exception("Test error")
        
#         # Setup executor to return our mock futures
#         mock_executor.return_value.__enter__.return_value.submit.side_effect = [
#             future1, future2
#         ]
        
#         result = self.collector.collect()
#         self.assertEqual(result.data, {"key1": "value1"})
#         self.assertTrue(len(result.error_handler.errors) > 0)


# if __name__ == '__main__':
#     unittest.main()