# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
from typing import Callable, Dict, Optional, Type
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase

KnowledgeType = Type[KnowledgeBase]


class KnowledgeFactory(object):
    _knowledge_pool: Dict[str, KnowledgeBase] = {}

    @classmethod
    def add_knowledge(cls, name, knowledge: KnowledgeBase):
        cls._knowledge_pool[name] = knowledge

    @classmethod
    def get_knowledge(cls, name) -> Optional[KnowledgeBase]:
        return cls._knowledge_pool.get(name)

    @classmethod
    def get_knowledge_pool(cls) -> Dict[str, KnowledgeBase]:
        return cls._knowledge_pool

    @classmethod
    def register(cls, name: str = '') -> Callable[[KnowledgeType], KnowledgeType]:
        def _deco(knowledge_cls: KnowledgeType) -> KnowledgeType:
            registered_name = name if name else knowledge_cls.__name__
            cls.add_knowledge(registered_name, knowledge_cls())
            return knowledge_cls
        return _deco
