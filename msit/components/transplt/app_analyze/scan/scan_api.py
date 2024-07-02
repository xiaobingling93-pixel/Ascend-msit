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

import traceback

from app_analyze.exception.source_scan_exception import \
    AutomakeExecuteFailedException, MakefileExecuteFailException, \
    SourceScanNoResultException, SourceFileNotFoundError
from app_analyze.common.kit_config import KitConfig, InputType, ScannerMode
from app_analyze.model.project import Project
from app_analyze.model.seq_project import SeqProject
from app_analyze.porting.input_factory import InputFactory
from app_analyze.utils.log_util import logger


class ScanApi:
    """
    :param api_flag: show the way of calling the api
        0: from command line
        1: from RESTFul interface
    """

    def __init__(self, api_flag=0, user_config_dict=None,
                 task_id=None, queue=None):
        self._api_flag = api_flag
        self._user_config_dict = user_config_dict
        self._task_id = task_id
        self._queue = queue
        self._worker = None

    @staticmethod
    def produce_report(project, not_empty=True):
        """
        功能：产生报告
        :return:
        """
        project.dump()
        project.generate_report(not_empty)

    @staticmethod
    def get_task_info(inputs):
        """
        获取任务基本信息
        :param inputs: 任务输入参数
        :return: task info
        """
        info = {"sourcedir": inputs.args.source,
                "constructtool": inputs.construct_tool
                }
        return info

    @staticmethod
    def _get_input_instance(input_type, param_dict):
        """ 调用inputs解析工厂获得结果 """
        inputs = InputFactory.get_input(input_type, param_dict)
        inputs.resolve_user_input()
        return inputs

    @staticmethod
    def _get_project_instance(inputs):
        if inputs.scanner_mode == ScannerMode.ALL.value:
            project = Project(inputs)
        else:
            project = SeqProject(inputs)
        return project

    def scan_source(self, param_dict):
        inputs, info = self._init_source_code_scan_task(param_dict)
        try:
            self._scan_source(inputs, info)
        except AutomakeExecuteFailedException as err:
            raise ValueError("{} msit transplt: error: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    err.get_info())) from err
        except MakefileExecuteFailException as err:
            raise ValueError("{} msit transplt: error: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    err.get_error_info())) from err
        except SourceScanNoResultException as err:
            raise ValueError("{} msit transplt: info: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    err.get_error_info())) from err
        except SourceFileNotFoundError as err:
            raise ValueError("{} msit transplt: info: {}".
                             format(KitConfig.PORTING_CONTENT,
                                    err.get_error_info())) from err
        except Exception as ex:
            logger.exception("The Scan task ended with error: %s.", ex)

    def _init_source_code_scan_task(self, param_dict):
        """ 初始化源码扫描任务的相关配置 """
        inputs = self._get_input_instance(InputType.CMD_LINE, param_dict)
        info = self.get_task_info(inputs)
        return inputs, info

    def _run_scan_source(self, project, info):
        """
        源码扫描过程执行
        """
        project.setup_reporters(info)
        project.setup_file_matrix()
        project.setup_scanners()

        try:
            project.scan()
        except FileNotFoundError as exp:
            raise SourceFileNotFoundError('source_file_not_found_err',
                                          'Source code not found') from exp

        if project.scan_results or project.lib_reports:
            self.produce_report(project)
            logger.info('**** Project analysis finished <<<')
        else:
            logger.info("There is nothing to be ported.")
            raise SourceScanNoResultException('source_scan_no_result_err',
                                              'There is nothing to be ported.')

    def _scan_source(self, inputs, info):
        """
        源码扫描整体过程控制
        """
        logger.info("Scan source files...")
        project = self._get_project_instance(inputs)
        try:
            self._run_scan_source(project, info)
        except AutomakeExecuteFailedException as err:
            self.produce_report(project, False)
            logger.info("**** AutomakeExecuteFailedException.")
            raise
        except MakefileExecuteFailException as err:
            self.produce_report(project, False)
            logger.info("**** MakefileExecuteFailException.")
            raise
        except SourceScanNoResultException as ex:
            # 源码扫描结果为空时捕获异常
            self.produce_report(project, False)
            logger.info("**** SourceScanNoResultException.")
            raise
        except SourceFileNotFoundError as ferr:
            self.produce_report(project, False)
            logger.info("**** SourceFileNotFoundError.")
            raise
        except TimeoutError as err:
            self.produce_report(project, False)
            logger.warning(err)
            logger.info('**** transplt project terminated <<<')
            logger.info('**** no error detected.')
        except KeyboardInterrupt as exp:
            logger.error('Keyboard Interrupted detected. Except:%s.', exp)
            logger.error('**** transplt project interrupted <<<')
        except Exception as ex:
            traceback.print_exc()
            logger.error(ex)
            logger.error('**** transplt project stopped <<<')
