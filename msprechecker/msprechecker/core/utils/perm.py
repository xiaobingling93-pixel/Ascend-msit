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

class FilePerm:
    def __init__(self, mode):
        self.mode = self._parse_mode(mode)
        if not (0 <= self.mode <= 0o777):
            raise ValueError("Mode must be between 0 and 0o777 (511)")

    def __eq__(self, other):
        return self.cmp_tuple() == self._other_bits(other)

    def __lt__(self, other):
        return self.cmp_tuple() < self._other_bits(other)

    def __le__(self, other):
        return self.cmp_tuple() <= self._other_bits(other)

    def __gt__(self, other):
        return self.cmp_tuple() > self._other_bits(other)

    def __ge__(self, other):
        return self.cmp_tuple() >= self._other_bits(other)
    
    def __str__(self):
        return oct(self.mode)

    def __repr__(self):
        return f"FilePerm({oct(self.mode)})>"
    
    @staticmethod
    def _parse_mode(mode):
        if isinstance(mode, int):
            return mode
        if isinstance(mode, str):
            return int(mode, 8)
        raise ValueError("Mode must be int or octal string")

    @staticmethod
    def _other_bits(other):
        if isinstance(other, FilePerm):
            return other.cmp_tuple()
        return FilePerm(other).cmp_tuple()

    def cmp_tuple(self):
        return tuple(int(b) for b in f"{self.mode:b}")
