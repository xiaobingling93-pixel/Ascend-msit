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

import unittest
from unittest.mock import patch
from components.debug.surgeon.auto_optimizer import SurgeonInstall
import pkg_resources

class TestSurgeonInstall(unittest.TestCase):

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='ais-bench')])
    def test_check_given_ais_bench_installed_when_check_then_return_ok(self):
        result = SurgeonInstall.check()
        self.assertEqual(result, "OK")

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='other-pkg')])
    def test_check_given_ais_bench_not_installed_when_check_then_return_warning(self):
        result = SurgeonInstall.check()
        expected_output = "[warnning] msit-benchmark not installed. will make the inference feature unusable. use `msit install benchmark` to try again"
        self.assertEqual(result, expected_output)

    @patch('pkg_resources.working_set', [])
    def test_check_given_no_packages_installed_when_check_then_return_warning(self):
        result = SurgeonInstall.check()
        expected_output = "[warnning] msit-benchmark not installed. will make the inference feature unusable. use `msit install benchmark` to try again"
        self.assertEqual(result, expected_output)

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='ais-bench'), pkg_resources.Distribution(project_name='other-pkg')])
    def test_check_given_multiple_packages_installed_when_check_then_return_ok(self):
        result = SurgeonInstall.check()
        self.assertEqual(result, "OK")

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='ais-bench'), pkg_resources.Distribution(project_name='ais-bench')])
    def test_check_given_ais_bench_installed_multiple_times_when_check_then_return_ok(self):
        result = SurgeonInstall.check()
        self.assertEqual(result, "OK")

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='other-pkg'), pkg_resources.Distribution(project_name='other-pkg')])
    def test_check_given_other_packages_installed_multiple_times_when_check_then_return_warning(self):
        result = SurgeonInstall.check()
        expected_output = "[warnning] msit-benchmark not installed. will make the inference feature unusable. use `msit install benchmark` to try again"
        self.assertEqual(result, expected_output)

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='ais-bench'), pkg_resources.Distribution(project_name='other-pkg'), pkg_resources.Distribution(project_name='ais-bench')])
    def test_check_given_ais_bench_and_other_packages_installed_multiple_times_when_check_then_return_ok(self):
        result = SurgeonInstall.check()
        self.assertEqual(result, "OK")

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='other-pkg'), pkg_resources.Distribution(project_name='ais-bench'), pkg_resources.Distribution(project_name='other-pkg')])
    def test_check_given_other_packages_and_ais_bench_installed_multiple_times_when_check_then_return_ok(self):
        result = SurgeonInstall.check()
        self.assertEqual(result, "OK")

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='other-pkg'), pkg_resources.Distribution(project_name='other-pkg'), pkg_resources.Distribution(project_name='other-pkg')])
    def test_check_given_only_other_packages_installed_multiple_times_when_check_then_return_warning(self):
        result = SurgeonInstall.check()
        expected_output = "[warnning] msit-benchmark not installed. will make the inference feature unusable. use `msit install benchmark` to try again"
        self.assertEqual(result, expected_output)

    @patch('pkg_resources.working_set', [pkg_resources.Distribution(project_name='ais-bench'), pkg_resources.Distribution(project_name='ais-bench'), pkg_resources.Distribution(project_name='ais-bench')])
    def test_check_given_ais_bench_installed_multiple_times_only_when_check_then_return_ok(self):
        result = SurgeonInstall.check()
        self.assertEqual(result, "OK")

if __name__ == '__main__':
    unittest.main()