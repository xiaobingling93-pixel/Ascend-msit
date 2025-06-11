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

import re
from functools import total_ordering

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError


def get_pkg_version(pkg_name):
    try:
        version_str = version(pkg_name)
    except PackageNotFoundError:
        try:
            from importlib import import_module
            version_str = import_module(pkg_name).__version__
        except Exception:
            version_str = None

    return version_str


@total_ordering
class Version:
    _regex = re.compile(
        r"^(?P<major>\d+)\.(?P<minor>\d+)"
        r"(?:\.(?P<patch>\d+))?"
        r"(?:\.rc(?P<rc>\d+))?"
        r"(?:\.b(?P<beta>\d+))?"
        r"(?:\.t(?P<test>\d+))?$"
    )

    def __init__(self, version_str):
        m = self._parse_version_str(version_str)
        self._major = int(m.group("major"))
        self._minor = int(m.group("minor"))
        self._patch = int(m.group("patch")) if m.group("patch") else None
        self._rc = int(m.group("rc")) if m.group("rc") else None
        self._beta = int(m.group("beta")) if m.group("beta") else None
        self._test = int(m.group("test")) if m.group("test") else None

    def __repr__(self):
        s = f"{self._major}.{self._minor}"
        if self._patch is not None:
            s += f".{self._patch}"
        if self._rc is not None:
            s += f".rc{self._rc}"
        if self._beta is not None:
            s += f".b{self._beta}"
        if self._test is not None:
            s += f".t{self._test}"
        return s

    def __eq__(self, other):
        return self.cmp_tuple() == self._other_version(other)

    def __lt__(self, other):
        return self.cmp_tuple() < self._other_version(other)

    def __le__(self, other):
        return self.cmp_tuple() <= self._other_version(other)

    def __gt__(self, other):
        return self.cmp_tuple() > self._other_version(other)

    def __ge__(self, other):
        return self.cmp_tuple() >= self._other_version(other)
    
    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def patch(self):
        return self._patch

    @property
    def rc(self):
        return self._rc

    @property
    def beta(self):
        return self._beta

    @property
    def test(self):
        return self._test
    
    @staticmethod
    def _parse_version_str(version_str):
        m = Version._regex.match(version_str)
        if not m:
            ver = get_pkg_version(version_str)
            if not ver:
                raise ValueError("Invalid version string: {}".format(version_str))

            return Version._parse_version_str(ver)
        return m
    
    @staticmethod
    def _other_version(other):
        if isinstance(other, Version):
            return other.cmp_tuple()
        return Version(other).cmp_tuple()

    def cmp_tuple(self):
        return (
            self._major,
            self._minor,
            self._patch if self._patch is not None else 0,
            self._rc if self._rc is not None else float("inf"),
            self._beta if self._beta is not None else float("inf")
        )
