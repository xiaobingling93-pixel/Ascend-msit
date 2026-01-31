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
import subprocess

from auto_optimizer.inference_engine.model_convert.compiler import Compiler
from components.debug.common import logger
from components.utils.util import filter_cmd
from components.utils.file_utils import check_file_or_directory_path


class OmCompiler(Compiler):

    def __init__(self, cfg):
        OmCompiler._check_required_params(cfg)
        cmd_type = cfg['type']
        self.atc_cmd = []
        if cmd_type == 'atc':
            self.atc_cmd.append('atc')
            for key, value in cfg.items():
                if key != 'type':
                    self.atc_cmd.append('--{}={}'.format(key, value))
        elif cmd_type == 'aoe':
            self.atc_cmd.append('aoe')
        else:
            raise RuntimeError("Invalid cmd type! Only support 'atc', 'aoe', but got '{}'.".format(cmd_type))

    @staticmethod
    def _check_required_params(cfg):
        required_params = ('type', 'framework', 'model', 'output', 'soc_version')
        params = cfg.keys()
        for param in required_params:
            if param not in params:
                raise RuntimeError("Parameter missing! '{}' is required in om convert!".format(param))

    def build_model(self):
        self.atc_cmd = filter_cmd(self.atc_cmd)
        logger.debug(self.atc_cmd)
        subprocess.run(self.atc_cmd, shell=False)


def onnx2om(path_onnx: str, converter: str, **kwargs):
    '''convert a onnx file to om using ATC.'''
    if not path_onnx.endswith('.onnx'):
        raise RuntimeError('Not a onnx file.')
    check_file_or_directory_path(path_onnx, is_strict=True)
    convert_cfg = {'type': converter, 'framework': '5', 'model': path_onnx, 'output': path_onnx[:-5], **kwargs}
    compiler = OmCompiler(convert_cfg)
    compiler.build_model()
    return path_onnx[:-4] + 'om'
