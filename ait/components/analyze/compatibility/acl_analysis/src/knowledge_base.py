# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

from knowledge import Knowledge, KnowledgeGroup


KnowledgeGroup.add_knowledge(
    Knowledge(
        "检查接口enginetype参数是否正确地使用了aclEngineType枚举，如果使用非法的枚举值则请修改代码并重新编译。",
        ['aclopCreateKernel'],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "aclmdlGetOutputNameByIndex接口返回信息格式有变更，如果涉及请重新进行atc模型转换，如果模型包含top名称则还需要适配返回值并重新编译。",
        ['aclmdlGetOutputNameByIndex'],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "请检查确保使用了aclrtCreateContext或aclrtSetCurrentContext接口显示设置context，否则请增加代码适配，并重新编译。",
        ['aclrtFreeHost'],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "检查aclgrphBuildModel接口是否有传入PRECISION_MODE、EXEC_DISABLE_REUSED_MEMORY、AUTO_TUNE_MODE三个参数，若没有，则无需处理；"
        "如果有，则检查下init中是否已配置，若未配置，建议在build接口的option中进行删除，逻辑跟原先一致，若已配置，需要用户根据需要进行配置修改。"
        "若全局仅生效一次，则只在init中配置即可，若每次build希望采用不同的option，则每次配置即可。"
        "另外检查aclgrphBuildModel接口INPUT_SHAPE参数设置，若涉及检查INPUT_SHAPE的传入值是否符合预期，如若不符请修改。",
        ['aclgrphBuildModel'],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge("检查aclcreateEvent接口创建的event资源是否超过了1023，若超过了会创建失败", ['aclcreateEvent'])
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "请检查参数是否有配置INPUT_FORMAT、INPUT_SHAPE、IS_DYNAMIC_INPUT、OP_NAME_MAP、OUTPUT_TYPE、LOG_LEVEL，如果有则需要删除这些参数。",
        ['aclgrphParseCaffe', 'aclgrphParseTensorFlow', 'aclgrphParseONNX', 'aclgrphParseONNXFromMem'],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "接口失败返回值-1在新版本已删除，增加细化了返回值，如果涉及使用返回值-1判断的情况，请修改代码适配。",
        [
            'CreateDvppApi',
            'DvppCtl',
            'DestroyDvppApi',
            'DvppGetOutParameter',
            'CreateVdecApi',
            'VdecCtl',
            'DestroyVdecApi',
            'CreateVenc',
            'SetVencParam',
            'RunVenc',
            'DestroyVenc',
        ],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "接口入参删除了aclrtStream，新版本不需要传递stream入参。",
        ['AclfvRepoAdd', 'AclfvRepoDel', 'AclfvDel', 'AclfvModify', 'AclfvSearch'],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "接口在新版本依赖python>=3.8及numpy>=1.22.0，若不想修改代码，则需要升级python和numpy，若要修改代码，则可用acl.util.bytes_to_ptr替代。",
        ['acl.util.numpy_to_ptr', 'acl.util.numpy_contiguous_to_ptr'],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "接口在新版本依赖python>=3.8及numpy>=1.22.0，若不想修改代码，则需要升级python和numpy，若要修改代码，则可用acl.util.ptr_to_bytes替代。",
        ['acl.util.ptr_to_numpy'],
    )
)


KnowledgeGroup.add_knowledge(
    Knowledge(
        "接口在新版本依赖python>=3.8及numpy>=1.22.0，若不想修改代码，则需要升级python和numpy，若要修改代码，接口的numpy输入可以替换成list输入。",
        [
            'acl.util.set_attr_list_int',
            'acl.util.set_attr_list_bool',
            'acl.util.set_attr_list_float',
            'acl.util.set_attr_list_list_int',
        ],
    )
)


KnowledgeGroup.add_knowledge(Knowledge("接口aclrtMallocHost在RC模型下需要替换成aclrtMalloc。", ['aclrtMallocHost']))


API_INPUT_MARCO = ['ACL_MEMCPY_HOST_TO_DEVICE', 'ACL_MEMCPY_DEVICE_TO_HOST']


def match_memcpy_type(line: str) -> bool:
    for marco in API_INPUT_MARCO:
        if marco in line:
            return True
    return False


KnowledgeGroup.add_knowledge(
    Knowledge(
        "接口输入参数kind在RC模型下不支持ACL_MEMCPY_DEVICE_TO_HOST和ACL_MEMCPY_HOST_TO_DEVICE，需要替换成ACL_MEMCPY_DEVICE_TO_DEVICE宏。",
        ['aclrtMemcpyAsync', 'aclrtMemcpy'],
        [match_memcpy_type],
    )
)


KnowledgeGroup.add_knowledge(Knowledge("接口aclrtFreeHost在RC模型下需要替换成aclrtFree。", ['aclrtFreeHost']))
