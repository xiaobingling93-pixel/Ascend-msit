# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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
def get_cmd_instance():
    from components.debug.compare import CompareCommand, BaseCommand
    from components.debug.surgeon import SurgeonCommand
    from components.debug.dump import DumpCommand

    compare_instance = CompareCommand("compare", "one-click network-wide accuracy analysis of golden models.")
    surgeon_instance = SurgeonCommand("surgeon", "surgeon tool for onnx modifying functions.")
    dump_instance = DumpCommand("dump", "msit debug dump")  # need to be fixed

    debug_sub_instances = [compare_instance, surgeon_instance, dump_instance]

    return BaseCommand("debug", "debug a wide variety of model issues.", debug_sub_instances)
