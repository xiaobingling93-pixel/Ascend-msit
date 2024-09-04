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
from components.utils.parser import BaseCommand


class SurgeonCommand(BaseCommand):
    def __init__(self, name, help_info) -> None:
        from components.debug.surgeon.auto_optimizer.ait_main import ListCommand, EvaluateCommand, OptimizeCommand, \
            ExtractCommand, ConcatenateCommand

        list_instance = ListCommand("list", "List avaiable Knowleges")
        evaluate_instance = EvaluateCommand("evaluate", "Evaluate model matching specified knowledges",
                                            alias_name="eva")
        optimize_instance = OptimizeCommand("optimize", "Optimize model with specified knowledges", alias_name="opt")
        extract_instance = ExtractCommand("extract", "Extract subgraph from onnx model", alias_name="ext")
        concatenate_instance = ConcatenateCommand("concatenate",
                                                  "Concatenate two onnxgraph into combined one onnxgraph",
                                                  alias_name="concat")

        surgeon_sub_instances = [
            list_instance,
            evaluate_instance,
            optimize_instance,
            extract_instance,
            concatenate_instance
        ]

        super().__init__(name, help_info, surgeon_sub_instances)