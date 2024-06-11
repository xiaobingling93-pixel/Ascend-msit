# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
import sys
import logging
import glob
import shutil

import aclruntime
import numpy as np
import pytest
from test_common import TestCommonClass

from ais_bench.infer.io_oprations import (
    create_pipeline_fileslist_from_inputs_list,
    PURE_INFER_FAKE_FILE_ZERO,
    PURE_INFER_FAKE_FILE_RANDOM,
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    @staticmethod
    def get_input_tensor_name():
        return "actual_input_1"

    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):
        logger.info('\n ---class level teardown_class')

    @classmethod
    def get_output_dir_bin(cls):
        return os.path.realpath(os.path.join(TestCommonClass.get_basepath(), cls.model_name, "output", "bin_out"))

    @classmethod
    def get_output_dir_npy(cls):
        return os.path.realpath(os.path.join(TestCommonClass.get_basepath(), cls.model_name, "output", "npy_out"))

    @classmethod
    def get_input_datas_file_bin_nor(cls):
        return os.path.realpath(
            os.path.join(TestCommonClass.get_basepath(), cls.model_name, "input", "fake_dataset_bin_nor/1.bin")
        )

    @classmethod
    def get_input_datas_dir_bin_nor(cls):
        return os.path.realpath(
            os.path.join(TestCommonClass.get_basepath(), cls.model_name, "input", "fake_dataset_bin_nor")
        )

    @classmethod
    def get_input_datas_file_bin_aipp(cls):
        return os.path.realpath(
            os.path.join(TestCommonClass.get_basepath(), cls.model_name, "input", "fake_dataset_bin_aipp/1.bin")
        )

    @classmethod
    def get_input_datas_dir_bin_aipp(cls):
        return os.path.realpath(
            os.path.join(TestCommonClass.get_basepath(), cls.model_name, "input", "fake_dataset_bin_aipp")
        )

    @classmethod
    def get_input_datas_file_npy_nor(cls):
        return os.path.realpath(
            os.path.join(TestCommonClass.get_basepath(), cls.model_name, "input", "fake_dataset_npy_nor/1.npy")
        )

    @classmethod
    def get_input_datas_dir_npy_nor(cls):
        return os.path.realpath(
            os.path.join(TestCommonClass.get_basepath(), cls.model_name, "input", "fake_dataset_npy_nor")
        )

    @classmethod
    def get_input_datas_file_npy_aipp(cls):
        return os.path.realpath(
            os.path.join(TestCommonClass.get_basepath(), cls.model_name, "input", "fake_dataset_npy_aipp/1.npy")
        )

    @classmethod
    def get_input_datas_dir_npy_aipp(cls):
        return os.path.realpath(
            os.path.join(TestCommonClass.get_basepath(), cls.model_name, "input", "fake_dataset_npy_aipp")
        )

    @classmethod
    def get_resnet_stcshape_om_path(cls, bs=1):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", f"pth_resnet50_bs{bs}.om")

    @classmethod
    def get_resnet_dymbatch_om_path(cls):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", "pth_resnet50_dymbatch.om")

    @classmethod
    def get_resnet_dymhw_om_path(cls):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", "pth_resnet50_dymwh.om")

    @classmethod
    def get_resnet_dymdim_om_path(cls):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", "pth_resnet50_dymdim.om")

    @classmethod
    def get_resnet_dymshape_om_path(cls):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", "pth_resnet50_dymshape.om")

    def init(self):
        self.model_name = "resnet50"

    def test_pure_infer_stc_batch_zero(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_stcshape_om_path(bs=1)
        session = aclruntime.InferenceSession(model_path, device_id, options)
        intensors_desc = session.get_inputs()
        infileslist = [[]]
        pure_file = PURE_INFER_FAKE_FILE_ZERO
        for _ in intensors_desc:
            infileslist[0].append(pure_file)
        infer_options = aclruntime.infer_options()
        infer_options.pure_infer_mode = True
        extra_session = []
        session.run_pipeline(infileslist, infer_options, extra_session)

    def test_pure_infer_stc_batch_random(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_stcshape_om_path(bs=1)
        session = aclruntime.InferenceSession(model_path, device_id, options)
        intensors_desc = session.get_inputs()
        infileslist = [[]]
        pure_file = PURE_INFER_FAKE_FILE_RANDOM
        for _ in intensors_desc:
            infileslist[0].append(pure_file)
        infer_options = aclruntime.infer_options()
        infer_options.pure_infer_mode = True
        extra_session = []
        session.run_pipeline(infileslist, infer_options, extra_session)

    def test_infer_stc_batch_input_file(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_stcshape_om_path(bs=1)
        session = aclruntime.InferenceSession(model_path, device_id, options)
        intensors_desc = session.get_inputs()
        infilespath = create_pipeline_fileslist_from_inputs_list(
            self.get_input_datas_file_bin_aipp().split(','), intensors_desc
        )
        infer_options = aclruntime.infer_options()
        extra_session = []
        session.run_pipeline(infilespath, infer_options, extra_session)

    def test_infer_stc_batch_input_file_out_bin(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_stcshape_om_path(bs=1)
        session = aclruntime.InferenceSession(model_path, device_id, options)
        intensors_desc = session.get_inputs()
        infilespath = create_pipeline_fileslist_from_inputs_list(
            self.get_input_datas_file_bin_aipp().split(','), intensors_desc
        )
        output_dir = self.get_output_dir_bin()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, 0o755)
        infer_options = aclruntime.infer_options()
        infer_options.output_dir = output_dir
        extra_session = []
        session.run_pipeline(infilespath, infer_options, extra_session)
        bin_files = glob.glob(os.path.join(output_dir, "*.bin"))
        assert len(bin_files) == 1

    def test_infer_stc_batch_input_file_out_npy(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_stcshape_om_path(bs=1)
        session = aclruntime.InferenceSession(model_path, device_id, options)
        intensors_desc = session.get_inputs()
        infilespath = create_pipeline_fileslist_from_inputs_list(
            self.get_input_datas_file_bin_aipp().split(','), intensors_desc
        )
        output_dir = self.get_output_dir_npy()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, 0o755)
        infer_options = aclruntime.infer_options()
        infer_options.output_dir = output_dir
        infer_options.out_format = 'NPY'
        extra_session = []
        session.run_pipeline(infilespath, infer_options, extra_session)
        npy_files = glob.glob(os.path.join(output_dir, "*.npy"))
        assert len(npy_files) == 1

    def test_infer_stc_batch_input_dir(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_stcshape_om_path(bs=1)
        session = aclruntime.InferenceSession(model_path, device_id, options)
        intensors_desc = session.get_inputs()
        infilespath = create_pipeline_fileslist_from_inputs_list(
            self.get_input_datas_dir_bin_aipp().split(','), intensors_desc
        )
        infer_options = aclruntime.infer_options()
        extra_session = []
        session.run_pipeline(infilespath, infer_options, extra_session)

    def test_infer_dym_batch_input_file(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_dymbatch_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_dynamic_batchsize(1)
        intensors_desc = session.get_inputs()
        infilespath = create_pipeline_fileslist_from_inputs_list(
            self.get_input_datas_file_bin_aipp().split(','), intensors_desc
        )
        infer_options = aclruntime.infer_options()
        extra_session = []
        session.run_pipeline(infilespath, infer_options, extra_session)

    def test_infer_dym_hw_input_file(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_dymhw_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_dynamic_hw(224, 224)
        intensors_desc = session.get_inputs()
        infilespath = create_pipeline_fileslist_from_inputs_list(
            self.get_input_datas_file_bin_nor().split(','), intensors_desc
        )
        infer_options = aclruntime.infer_options()
        extra_session = []
        session.run_pipeline(infilespath, infer_options, extra_session)

    def test_infer_dym_dim_input_file(self):
        device_id = 0
        input_tensor_name = self.get_input_tensor_name()
        options = aclruntime.session_options()
        model_path = self.get_resnet_dymdim_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_dynamic_dims(input_tensor_name + ":1,3,224,224")
        intensors_desc = session.get_inputs()
        infilespath = create_pipeline_fileslist_from_inputs_list(
            self.get_input_datas_file_bin_nor().split(','), intensors_desc
        )
        infer_options = aclruntime.infer_options()
        extra_session = []
        session.run_pipeline(infilespath, infer_options, extra_session)

    def test_infer_auto_dim_input_file(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_dymdim_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        intensors_desc = session.get_inputs()
        infilespath = create_pipeline_fileslist_from_inputs_list(
            self.get_input_datas_file_npy_nor().split(','), intensors_desc
        )
        infer_options = aclruntime.infer_options()
        infer_options.auto_dym_dims = True
        extra_session = []
        session.run_pipeline(infilespath, infer_options, extra_session)

    def test_infer_intensor_infile_not_matched(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_resnet_stcshape_om_path(bs=2)
        session = aclruntime.InferenceSession(model_path, device_id, options)
        intensors_desc = session.get_inputs()
        with pytest.raises(RuntimeError) as e:
            infilespath = create_pipeline_fileslist_from_inputs_list(
                self.get_input_datas_file_bin_aipp().split(','), intensors_desc
            )
