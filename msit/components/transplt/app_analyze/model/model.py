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

import os
import argparse
from copy import deepcopy

from app_analyze.model.seq_project import SeqProject
from app_analyze.porting.input_factory import InputFactory
from app_analyze.common.kit_config import InputType, KitConfig
from app_analyze.scan.sequence.seq_handler import filter_api_seqs
from app_analyze.scan.sequence.seq_desc import get_idx_tbl, set_api_lut
from app_analyze.scan.sequence.acc_libs import set_expert_libs, get_expert_libs, expert_libs_to_dict
from app_analyze.utils.io_util import IOUtil
from app_analyze.utils import log_util
from app_analyze.__main__ import get_cmd_instance
from components.utils.parser import BaseCommand


class LoadSequencesCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--seqs", required=True, help="seqs results file, saved by api ids")
        parser.add_argument("--seqs-idx", required=True, help="id and seq mapping file")
        parser.add_argument(
            "--log-level", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="specify log level"
        )

    def handle(self, args):
        log_util.set_logger_level(args.log_level)
        log_util.init_file_logger()
        return Model.clean_seqs(args.seqs, args.seqs_idx)


class Model:
    @staticmethod
    def _load_data(path):
        if not os.path.exists(path):
            raise Exception(f'{path} is not existed!')

        rst = list()
        for item in os.scandir(path):
            if item.is_dir():
                rst.append(item.path)
            elif item.is_file():
                rst.append(item.path)
        return rst

    @staticmethod
    def _scan_sources(files, args):
        api_seqs = list()
        new_args = deepcopy(args)
        for path in files:
            new_args.source = path
            inputs = InputFactory.get_input(InputType.CUSTOM, new_args)
            inputs.resolve_user_input()

            project = SeqProject(inputs, False)
            project.setup_file_matrix()
            project.setup_scanners()
            project.scan()
            api_seqs += project.get_api_seqs()

        return api_seqs

    def train(self, args):
        args.source = os.path.abspath(args.source)
        if not os.path.exists(args.source):
            raise Exception(f'Source directory {args.source} is not existed!')

        KitConfig.SOURCE_DIRECTORY = args.source

        log_util.set_logger_level(args.log_level)
        log_util.init_file_logger()
        set_api_lut()

        dataset = self._load_data(args.source)
        api_seqs = self._scan_sources(dataset, args)

        seqs = filter_api_seqs(api_seqs)
        idx_seq_dict = get_idx_tbl()
        return api_seqs, seqs, idx_seq_dict

    @staticmethod
    def clean_seqs(seqs_file, idx_seq_file):
        idx_seq_dict = IOUtil.json_safe_load(idx_seq_file)
        set_api_lut(idx_seq_dict)

        api_seqs = IOUtil.json_safe_load(seqs_file)
        seqs = filter_api_seqs(api_seqs, idx_seq_dict)
        return api_seqs, seqs, idx_seq_dict

    @staticmethod
    def export_expert_libs(path='./'):
        all_idx_dict = dict()
        for val in KitConfig.API_INDEX_MAP.values():
            idx_seq_dict = IOUtil.json_safe_load(val)
            all_idx_dict.update(idx_seq_dict)
        set_api_lut(all_idx_dict)

        expert_libs = IOUtil.json_safe_load(KitConfig.EXPERT_LIBS_FILE)
        set_expert_libs(expert_libs)

        expert_libs = get_expert_libs()
        rs_dict = expert_libs_to_dict(expert_libs)

        file = path + 'expert_libs_debug.json'
        IOUtil.json_safe_dump(rs_dict, file)


def predict():
    instance = get_cmd_instance()
    parser = argparse.ArgumentParser()
    instance.add_arguments(parser)
    args = parser.parse_args()
    instance.handle(args)


def train():
    instance = get_cmd_instance()
    parser = argparse.ArgumentParser()
    instance.add_arguments(parser)
    args = parser.parse_args()

    model = Model()
    model.train(args)


def postprocess():
    help_info = "Process sequences to identify frequent sub-sequences"
    instance = LoadSequencesCommand("postprocess", help_info)
    parser = argparse.ArgumentParser()
    instance.add_arguments(parser)
    args = parser.parse_args()
    instance.handle(args)
