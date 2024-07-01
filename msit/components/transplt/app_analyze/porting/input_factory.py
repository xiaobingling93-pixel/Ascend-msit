# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

from app_analyze.common.kit_config import InputType
from app_analyze.porting.cmdline_input import CommandLineInput
from app_analyze.porting.custom_input import CustomInput


class InputFactory:
    """
    输入工厂类，根据输入的类型返回具体的输入子类对象
    """

    @staticmethod
    def get_input(input_type, args):
        """
        根据输入类型实例化对应的子类对象返回,目前仅规划了命令行的输入，如果后续需要扩展，添加类型和else分支.
        :param input_type: 输入类型
        :param args: 输入的初始化信息
        :return: 子类对象
        """
        if input_type == InputType.CMD_LINE:
            return CommandLineInput(args)
        elif input_type == InputType.CUSTOM:
            return CustomInput(args)
        else:
            raise Exception('Not support yet!')
