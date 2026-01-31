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
import unittest
from components.profile.msit_prof.msprof.args_adapter import MsProfArgsAdapter

class TestMsProfArgsAdapter(unittest.TestCase):
    def test_msprofargsadapter_initialization(self):
        # 准备测试数据
        application = "test_app"
        output = "test_output"
        model_execution = True
        sys_hardware_mem = False
        sys_cpu_profiling = True
        sys_profiling = False
        sys_pid_profiling = True
        dvpp_profiling = False
        runtime_api = "test_runtime_api"
        task_time = 100
        aicpu = True

        # 创建 MsProfArgsAdapter 实例
        msprof_args_adapter = MsProfArgsAdapter(
            application, output, model_execution, sys_hardware_mem, sys_cpu_profiling, sys_profiling,
            sys_pid_profiling, dvpp_profiling, runtime_api, task_time, aicpu
        )

        # 验证属性是否正确赋值
        self.assertEqual(msprof_args_adapter.application, application)
        self.assertEqual(msprof_args_adapter.output, output)
        self.assertEqual(msprof_args_adapter.model_execution, model_execution)
        self.assertEqual(msprof_args_adapter.sys_hardware_mem, sys_hardware_mem)
        self.assertEqual(msprof_args_adapter.sys_cpu_profiling, sys_cpu_profiling)
        self.assertEqual(msprof_args_adapter.sys_profiling, sys_profiling)
        self.assertEqual(msprof_args_adapter.sys_pid_profiling, sys_pid_profiling)
        self.assertEqual(msprof_args_adapter.dvpp_profiling, dvpp_profiling)
        self.assertEqual(msprof_args_adapter.runtime_api, runtime_api)
        self.assertEqual(msprof_args_adapter.task_time, task_time)
        self.assertEqual(msprof_args_adapter.aicpu, aicpu)

if __name__ == '__main__':
    unittest.main()