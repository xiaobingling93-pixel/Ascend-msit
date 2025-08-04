# -*- coding: utf-8 -*-
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

import os
import json
import argparse

from msguard.security import open_s

from .base import CommandType
from .legacy import show_legacy_warnings
from .banner import BannerPresenter
from ..collectors import ConfigCollector
from ..reporters import Reporter
from ..presets import RuleManager
from .base import CommandStrategy, CommandType
from ..utils import CheckErrorHandler, ConfigErrorHandler, global_logger
from ..collectors import (
    EnvCollector, SysCollector, ConfigCollector,
    AscendCollector, HCCLCollector, PingCollector,
    WeightCollector, CPUStressCollector, NPUStressCollector,
    UserConfigCollector, MindIEEnvCollector, ModelConfigCollector
)
from ..checkers import (
    UserConfigChecker, MindIEEnvChecker, 
    ModelConfigChecker, EnvChecker, SysChecker,
    AscendChecker, HCCLChecker, StressChecker
)
from ..comparators import Comparator
from ..reporters import Reporter


class CollectorFactory:
    @staticmethod
    def create(args: argparse.Namespace):
        collectors = [
            EnvCollector(filter_env=getattr(args, 'filter', False)),
            SysCollector(),
            AscendCollector(),
            HCCLCollector(),
        ]
        
        if getattr(args, "hardware", False):
            collectors.extend((CPUStressCollector(), NPUStressCollector()))

        if getattr(args, "user_config_path", None):
            collectors.append(UserConfigCollector(config_path=args.user_config_path))

        if getattr(args, "mindie_env_path", None):
            collectors.append(MindIEEnvCollector(config_path=args.mindie_env_path))

        if getattr(args, 'rank_table_path', None):
            collectors.append(PingCollector(rank_table_file=args.rank_table_path))

        if getattr(args, "weight_dir", None):
            model_config_path = os.path.join(args.weight_dir, "config.json")
            collectors.append(ModelConfigCollector(config_path=model_config_path))

            if args.command == CommandType.CMD_DUMP and getattr(args, 'chunk_size', None):
                collectors.append(WeightCollector(weight_dir=args.weight_dir, chunk_size=args.chunk_size * 1024))

        return collectors


class CheckerFactory:
    """Checker工厂类"""
    _registry = {
        EnvCollector: EnvChecker,
        SysCollector: SysChecker,
        AscendCollector: AscendChecker,
        HCCLCollector: HCCLChecker,
        CPUStressCollector: StressChecker,
        NPUStressCollector: StressChecker,
        UserConfigCollector: UserConfigChecker,
        MindIEEnvCollector: MindIEEnvChecker,
        ModelConfigCollector: ModelConfigChecker
    }
    
    @classmethod
    def register(cls, collector_class, checker_class) -> None:
        cls._registry[collector_class] = checker_class
    
    @classmethod
    def create(cls, collector_class):
        """根据collector类型创建对应的checker"""
        if collector_class not in cls._registry:
            raise KeyError(f"No checker registered for collector: {collector_class.__name__}")
        
        return cls._registry[collector_class]


class PrecheckStrategy(CommandStrategy):
    @staticmethod
    def execute(args: argparse.Namespace) -> int:
        rule_manager = RuleManager(args.custom_config_path)
        reporter = Reporter()
        
        collectors = CollectorFactory.create(args)
        for collector in collectors:
            collect_result = collector.collect()
            if not collect_result.error_handler.empty():
                reporter.report(collect_result.error_handler)
                continue
            
            data = collect_result.data
            error_handler = CheckErrorHandler(severity=args.severity_level)
            
            if isinstance(collector, ConfigCollector):
                data, file_lines, key_mapping, context_hierarchy = collect_result.data
                error_handler = ConfigErrorHandler(
                    args.severity_level, 
                    file_lines, 
                    key_mapping, 
                    context_hierarchy
                )
            
            checker = CheckerFactory.create(collector.__class__)(
                rule_manager=rule_manager, 
                error_handler=error_handler
            )
            reporter.report(checker.check(data))

        return 0


class DumpStrategy(CommandStrategy):
    @staticmethod
    def execute(args: argparse.Namespace) -> int:
        collectors = CollectorFactory.create(args)

        dump_content = {}
        for collector in collectors:
            collect_result = collector.collect()
            if not collect_result.error_handler.empty():
                DumpStrategy._display_collect_warning(collect_result.error_handler)
                continue

            collect_type = collect_result.error_handler.type
            data = collect_result.data

            if isinstance(collector, ConfigCollector):
                data, _, _, _ = collect_result.data
            
            dump_content[collect_type] = data

        with open_s(args.output_path, "w") as f:
            json.dump(dump_content, f, indent=4)
            
            global_logger.info(
                "All information has been saved in: %r. You can use '--output-path' to specify the save location.", 
                args.output_path
            )
            global_logger.info(
                "What's Next?\n\tYou may now use 'msprechecker compare' to compare two or more dumped files for discrepancies"
            )
    
    @staticmethod
    def _display_collect_warning(error_handler):
        for error in error_handler:
            global_logger.warning(
                "Error occured while collecting '%s': %s", 
                error_handler.type, 
                error.what
            )


class CompareStrategy(CommandStrategy):
    @staticmethod
    def execute(args: argparse.Namespace) -> int:
        if len(args.dumped_path) < 2:
            global_logger.error("You need two or more files to compare!")
            return 1
        
        path_to_data = CompareStrategy._load_dumped_files(args.dumped_path)
        reporter = Reporter()

        comparator = Comparator()
        reporter.report(comparator.compare(path_to_data))


    @staticmethod
    def _load_dumped_files(file_paths):
        """Load dumped JSON files for comparison"""
        path_to_data = {}

        for path in file_paths:
            with open_s(path) as f:
                path_to_data[path] = json.load(f)

        return path_to_data


class CommandStrategyFactory:    
    def __init__(self) -> None:
        self._registry = {
            CommandType.CMD_PRECHECK: PrecheckStrategy,
            CommandType.CMD_DUMP: DumpStrategy,
            CommandType.CMD_COMPARE: CompareStrategy
        }

    def register(self, cmd_type, strategy_class) -> None:
        self._registry[cmd_type] = strategy_class

    def create_strategy(self, cmd: CommandType) -> CommandStrategy:
        if cmd not in self._registry:
            raise ValueError(f"No strategy registered for command: {cmd}")
        
        return self._registry[cmd]()


class Coordinator:
    def __init__(self) -> None:
        self._strategy_factory = CommandStrategyFactory()

    def execute(self, parser: argparse.ArgumentParser) -> int:
        """Execute the appropriate action based on command"""
        args = parser.parse_args()
        show_legacy_warnings(args)
        
        cmd = getattr(args, 'command', None)
        if not cmd:
            BannerPresenter().print_banner()
            parser.print_help()
            return 1
        
        args.command = CommandType(cmd)
        strategy = self._strategy_factory.create_strategy(args.command)
        return strategy.execute(args)
