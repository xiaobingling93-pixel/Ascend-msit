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
from msprechecker.core.utils.version import Version, get_pkg_version


class TestVersion(unittest.TestCase):
    def test_parse_basic_version(self):
        v = Version("1.2")
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 2)
        self.assertIsNone(v.patch)
        self.assertIsNone(v.rc)
        self.assertIsNone(v.beta)
        self.assertIsNone(v.test)

    def test_parse_full_version(self):
        v = Version("1.2.3.rc4.b5.t6")
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 2)
        self.assertEqual(v.patch, 3)
        self.assertEqual(v.rc, 4)
        self.assertEqual(v.beta, 5)
        self.assertEqual(v.test, 6)

    def test_parse_partial_version(self):
        v = Version("2.5.7.rc3")
        self.assertEqual(v.major, 2)
        self.assertEqual(v.minor, 5)
        self.assertEqual(v.patch, 7)
        self.assertEqual(v.rc, 3)
        self.assertIsNone(v.beta)
        self.assertIsNone(v.test)

    def test_parse_invalid_version(self):
        with self.assertRaises(ValueError):
            Version("invalid.version")

    def test_repr_basic(self):
        self.assertEqual(str(Version("1.2")), "1.2")

    def test_repr_full(self):
        self.assertEqual(str(Version("1.2.3.rc4.b5.t6")), "1.2.3.rc4.b5.t6")

    def test_repr_partial(self):
        self.assertEqual(str(Version("3.4.5.rc6")), "3.4.5.rc6")

    def test_equality(self):
        self.assertEqual(Version("1.2"), Version("1.2"))
        self.assertEqual(Version("1.2.3"), Version("1.2.3"))
        self.assertEqual(Version("1.2.3.rc4"), Version("1.2.3.rc4"))
        self.assertEqual(Version("8.0"), Version("8.0.0"))

        self.assertNotEqual(Version("1.2"), Version("1.3"))
        self.assertNotEqual(Version("1.2.3"), Version("1.2.4"))

    def test_comparison(self):
        self.assertLess(Version("1.2"), Version("1.3"))

        self.assertLess(Version("1.2"), Version("1.2.1"))
        self.assertLess(Version("1.2.3.rc1"), Version("1.2.3.rc2"))

        self.assertGreater(Version("1.2.1"), Version("1.2"))
        self.assertGreater(Version("1.2.3"), Version("1.2.3.rc1"))
        self.assertGreater(Version("2.0"), Version("1.9"))
        self.assertGreater(Version("1.2.3.rc2"), Version("1.2.3.rc1"))
        self.assertGreater(Version("1.2.3.rc1"), Version("1.2.3.rc1.b1"))

    def test_comparison_with_strings(self):
        self.assertLess(Version("1.2"), "1.3")
        self.assertLess("1.2", Version("1.3"))
        self.assertEqual(Version("1.2"), "1.2")
        self.assertEqual("1.2", Version("1.2"))

    def test_cmp_tuple(self):
        inf = float('inf')
        self.assertEqual(Version("1.2").cmp_tuple(), (1, 2, 0, inf, inf))
        self.assertEqual(Version("1.2.3").cmp_tuple(), (1, 2, 3, inf, inf))
        self.assertEqual(Version("1.2.3.rc4").cmp_tuple(), (1, 2, 3, 4, inf))
        self.assertEqual(Version("1.2.3.rc4.b5").cmp_tuple(), (1, 2, 3, 4, 5))

    def test_parse_version_str_fallback(self):
        with patch('msprechecker.core.utils.version.get_pkg_version', return_value="1.2.3"):
            v = Version("some_package")
            self.assertEqual(v.major, 1)
            self.assertEqual(v.minor, 2)
            self.assertEqual(v.patch, 3)

    def test_parse_version_str_fallback_failure(self):
        with patch('msprechecker.core.utils.version.get_pkg_version', return_value=None):
            with self.assertRaises(ValueError):
                Version("some_package")


class TestGetPkgVersion(unittest.TestCase):
    @patch('importlib.import_module')
    def test_get_pkg_version_fallback(self, mock_import):
        mock_import.return_value.__version__ = "1.2.3"
        self.assertEqual(get_pkg_version("some_package"), "1.2.3")

    def test_get_pkg_version_failure(self):
        self.assertIsNone(get_pkg_version("some_package"))

    @patch('msprechecker.core.utils.version.version', return_value="1.2.3")
    def test_get_pkg_version_success(self, mock_version):
        self.assertEqual(get_pkg_version("some_package"), "1.2.3")
