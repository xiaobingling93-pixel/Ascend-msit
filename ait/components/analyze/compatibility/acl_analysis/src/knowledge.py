# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

from typing import List, Callable

MatchFunc = Callable[[str], bool]


class Knowledge(object):

    def __init__(self, suggestion: str, apis: List[str] = None, match_funcs: List[MatchFunc] = None) -> None:
        apis = apis or []
        match_funcs = match_funcs or []
        self._suggestion: str = suggestion
        self._apis = apis
        self._match_funcs = match_funcs

    @property
    def suggestion(self):
        return self._suggestion

    @property
    def apis(self):
        return self._apis

    def analysis(self, line: str) -> bool:
        for func in self._match_funcs:
            if not func(line):
                return False
        return True


class KnowledgeGroup:
    _knowledges: List[Knowledge] = []

    @classmethod
    def add_knowledge(cls, knowledge: Knowledge):
        cls._knowledges.append(knowledge)

    @classmethod
    def get_knowledges(cls) -> List[Knowledge]:
        return cls._knowledges
