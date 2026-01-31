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
class MsProfArgsAdapter():
    def __init__(self, application, output, model_execution, sys_hardware_mem, sys_cpu_profiling, sys_profiling,
                 sys_pid_profiling, dvpp_profiling, runtime_api, task_time, aicpu):
        self.application = application
        self.output = output
        self.model_execution = model_execution
        self.sys_hardware_mem = sys_hardware_mem
        self.sys_cpu_profiling = sys_cpu_profiling
        self.sys_profiling = sys_profiling
        self.sys_pid_profiling = sys_pid_profiling
        self.dvpp_profiling = dvpp_profiling
        self.runtime_api = runtime_api
        self.task_time = task_time
        self.aicpu = aicpu