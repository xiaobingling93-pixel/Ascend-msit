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

import logging
import os
from configparser import ConfigParser
import aclruntime
import numpy as np
import pytest
from test_common import TestCommonClass

logger = logging.getLogger(__name__)


class TestClass:
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
    def get_input_tensor_name(cls):
        return "actual_input_1"

    @classmethod
    def load_aipp_config_file(cls, session, config_file, batchsize):
        cfg = ConfigParser()
        cfg.read(config_file, 'UTF-8')
        session_list = cfg.sections()
        # 多个aipp输入不支持
        if session_list.count('aipp_op') != 1:
            raise RuntimeError('wrong aipp config file content!')
        option_list = cfg.options('aipp_op')
        if option_list.count('input_format') == 1:
            cls.aipp_set_input_format(cfg, session)
        else:
            raise RuntimeError('wrong aipp config file content!')

        if option_list.count('src_image_size_w') == 1 and option_list.count('src_image_size_h') == 1:
            cls.aipp_set_src_image_size(cfg, session)
        else:
            raise RuntimeError('wrong aipp config file content!')
        session.aipp_set_max_batch_size(batchsize)
        try:
            cls.aipp_set_rbuv_swap_switch(cfg, session, option_list)
            cls.aipp_set_ax_swap_switch(cfg, session, option_list)
            cls.aipp_set_csc_params(cfg, session, option_list)
            cls.aipp_set_crop_params(cfg, session, option_list)
            cls.aipp_set_padding_params(cfg, session, option_list)
            cls.aipp_set_dtc_pixel_mean(cfg, session, option_list)
            cls.aipp_set_dtc_pixel_min(cfg, session, option_list)
            cls.aipp_set_pixel_var_reci(cfg, session, option_list)
        except RuntimeError as err:
            raise RuntimeError("params illegal") from err
        ret = session.set_dym_aipp_info_set()
        return ret

    @classmethod
    def aipp_set_input_format(cls, cfg, session):
        input_format = cfg.get('aipp_op', 'input_format')
        legal_format = ["YUV420SP_U8", "XRGB8888_U8", "RGB888_U8", "YUV400_U8"]
        if legal_format.count(input_format) == 1:
            session.aipp_set_input_format(input_format)
        else:
            raise RuntimeError('wrong aipp config file content!')

    @classmethod
    def aipp_set_src_image_size(cls, cfg, session):
        src_image_size = list()
        tmp_size_w = cfg.getint('aipp_op', 'src_image_size_w')
        tmp_size_h = cfg.getint('aipp_op', 'src_image_size_h')
        if 2 <= tmp_size_w <= 4096:
            src_image_size.append(tmp_size_w)
        else:
            raise RuntimeError('wrong aipp config file content!')
        if 1 <= tmp_size_h <= 4096:
            src_image_size.append(tmp_size_h)
        else:
            raise RuntimeError('wrong aipp config file content!')

        session.aipp_set_src_image_size(src_image_size)

    @classmethod
    def aipp_set_rbuv_swap_switch(cls, cfg, session, option_list):
        if option_list.count('rbuv_swap_switch') == 0:
            session.aipp_set_rbuv_swap_switch(0)
            return
        tmp_rs_switch = cfg.getint('aipp_op', 'rbuv_swap_switch')
        if tmp_rs_switch == 0 or tmp_rs_switch == 1:
            session.aipp_set_rbuv_swap_switch(tmp_rs_switch)
        else:
            raise RuntimeError('wrong aipp config file content!')

    @classmethod
    def aipp_set_ax_swap_switch(cls, cfg, session, option_list):
        if option_list.count('ax_swap_switch') == 0:
            session.aipp_set_ax_swap_switch(0)
            return
        tmp_as_switch = cfg.getint('aipp_op', 'ax_swap_switch')
        if tmp_as_switch == 0 or tmp_as_switch == 1:
            session.aipp_set_ax_swap_switch(tmp_as_switch)
        else:
            raise RuntimeError('wrong aipp config file content!')

    @classmethod
    def aipp_set_csc_params(cls, cfg, session, option_list):
        if option_list.count('csc_switch') == 0:
            tmp_csc_switch = 0
        else:
            tmp_csc_switch = cfg.getint('aipp_op', 'csc_switch')

        if tmp_csc_switch == 0:
            tmp_csc_params = [0] * 16
        elif tmp_csc_switch == 1:
            tmp_csc_params = list()
            tmp_csc_params.append(tmp_csc_switch)
            tmp_csc_params.append(0 if option_list.count('matrix_r0c0') == 0 else cfg.getint('aipp_op', 'matrix_r0c0'))
            tmp_csc_params.append(0 if option_list.count('matrix_r0c1') == 0 else cfg.getint('aipp_op', 'matrix_r0c1'))
            tmp_csc_params.append(0 if option_list.count('matrix_r0c2') == 0 else cfg.getint('aipp_op', 'matrix_r0c2'))
            tmp_csc_params.append(0 if option_list.count('matrix_r1c0') == 0 else cfg.getint('aipp_op', 'matrix_r1c0'))
            tmp_csc_params.append(0 if option_list.count('matrix_r1c1') == 0 else cfg.getint('aipp_op', 'matrix_r1c1'))
            tmp_csc_params.append(0 if option_list.count('matrix_r1c2') == 0 else cfg.getint('aipp_op', 'matrix_r1c2'))
            tmp_csc_params.append(0 if option_list.count('matrix_r2c0') == 0 else cfg.getint('aipp_op', 'matrix_r2c0'))
            tmp_csc_params.append(0 if option_list.count('matrix_r2c1') == 0 else cfg.getint('aipp_op', 'matrix_r2c1'))
            tmp_csc_params.append(0 if option_list.count('matrix_r2c2') == 0 else cfg.getint('aipp_op', 'matrix_r2c2'))
            tmp_csc_params.append(
                0 if option_list.count('output_bias_0') == 0 else cfg.getint('aipp_op', 'output_bias_0')
            )
            tmp_csc_params.append(
                0 if option_list.count('output_bias_1') == 0 else cfg.getint('aipp_op', 'output_bias_1')
            )
            tmp_csc_params.append(
                0 if option_list.count('output_bias_2') == 0 else cfg.getint('aipp_op', 'output_bias_2')
            )
            tmp_csc_params.append(
                0 if option_list.count('input_bias_0') == 0 else cfg.getint('aipp_op', 'input_bias_0')
            )
            tmp_csc_params.append(
                0 if option_list.count('input_bias_1') == 0 else cfg.getint('aipp_op', 'input_bias_1')
            )
            tmp_csc_params.append(
                0 if option_list.count('input_bias_2') == 0 else cfg.getint('aipp_op', 'input_bias_2')
            )

            range_ok = True
            for i in range(1, 9):
                range_ok = range_ok and (-32677 <= tmp_csc_params[i] <= 32676)
            for i in range(10, 15):
                range_ok = range_ok and (0 <= tmp_csc_params[i] <= 255)
            if range_ok is False:
                raise RuntimeError('wrong aipp config file content!')
        else:
            raise RuntimeError('wrong aipp config file content!')

        session.aipp_set_csc_params(tmp_csc_params)

    @classmethod
    def aipp_set_crop_params(cls, cfg, session, option_list):
        if option_list.count('crop') == 0:
            tmp_crop_switch = 0
        else:
            tmp_crop_switch = cfg.getint('aipp_op', 'crop')

        if tmp_crop_switch == 0:
            tmp_crop_params = [0, 0, 0, 416, 416]
        elif tmp_crop_switch == 1:
            tmp_crop_params = list()
            tmp_crop_params.append(tmp_crop_switch)
            tmp_crop_params.append(
                0 if option_list.count('load_start_pos_w') == 0 else cfg.getint('aipp_op', 'load_start_pos_w')
            )
            tmp_crop_params.append(
                0 if option_list.count('load_start_pos_h') == 0 else cfg.getint('aipp_op', 'load_start_pos_h')
            )
            tmp_crop_params.append(0 if option_list.count('crop_size_w') == 0 else cfg.getint('aipp_op', 'crop_size_w'))
            tmp_crop_params.append(0 if option_list.count('crop_size_h') == 0 else cfg.getint('aipp_op', 'crop_size_h'))

            range_ok = True
            range_ok = range_ok and (0 <= tmp_crop_params[1] <= 4095)
            range_ok = range_ok and (0 <= tmp_crop_params[2] <= 4095)
            range_ok = range_ok and (1 <= tmp_crop_params[3] <= 4096)
            range_ok = range_ok and (1 <= tmp_crop_params[4] <= 4096)
            if range_ok is False:
                raise RuntimeError('wrong aipp config file content!')
        else:
            raise RuntimeError('wrong aipp config file content!')

        session.aipp_set_crop_params(tmp_crop_params)

    @classmethod
    def aipp_set_padding_params(cls, cfg, session, option_list):
        if option_list.count('padding') == 0:
            tmp_padding_switch = 0
        else:
            tmp_padding_switch = cfg.getint('aipp_op', 'padding')

        if tmp_padding_switch == 0:
            tmp_padding_params = [0] * 5
        elif tmp_padding_switch == 1:
            tmp_padding_params = list()
            tmp_padding_params.append(tmp_padding_switch)
            tmp_padding_params.append(
                0 if option_list.count('padding_size_top') == 0 else cfg.getint('aipp_op', 'padding_size_top')
            )
            tmp_padding_params.append(
                0 if option_list.count('padding_size_bottom') == 0 else cfg.getint('aipp_op', 'padding_size_bottom')
            )
            tmp_padding_params.append(
                0 if option_list.count('padding_size_left') == 0 else cfg.getint('aipp_op', 'padding_size_left')
            )
            tmp_padding_params.append(
                0 if option_list.count('padding_size_right') == 0 else cfg.getint('aipp_op', 'padding_size_right')
            )

            range_ok = True
            for i in range(1, 5):
                range_ok = range_ok and (0 <= tmp_padding_params[i] <= 32)
            if range_ok is False:
                raise RuntimeError('wrong aipp config file content!')
        else:
            raise RuntimeError('wrong aipp config file content!')

        session.aipp_set_padding_params(tmp_padding_params)

    @classmethod
    def aipp_set_dtc_pixel_mean(cls, cfg, session, option_list):
        tmp_mean_params = list()
        tmp_mean_params.append(0 if option_list.count('mean_chn_0') == 0 else cfg.getint('aipp_op', 'mean_chn_0'))
        tmp_mean_params.append(0 if option_list.count('mean_chn_1') == 0 else cfg.getint('aipp_op', 'mean_chn_1'))
        tmp_mean_params.append(0 if option_list.count('mean_chn_2') == 0 else cfg.getint('aipp_op', 'mean_chn_2'))
        tmp_mean_params.append(0 if option_list.count('mean_chn_3') == 0 else cfg.getint('aipp_op', 'mean_chn_3'))

        range_ok = True
        for i in range(0, 4):
            range_ok = range_ok and (0 <= tmp_mean_params[i] <= 255)
        if range_ok is False:
            raise RuntimeError('wrong aipp config file content!')

        session.aipp_set_dtc_pixel_mean(tmp_mean_params)

    @classmethod
    def aipp_set_dtc_pixel_min(cls, cfg, session, option_list):
        tmp_min_params = list()
        tmp_min_params.append(0 if option_list.count('min_chn_0') == 0 else cfg.getfloat('aipp_op', 'min_chn_0'))
        tmp_min_params.append(0 if option_list.count('min_chn_1') == 0 else cfg.getfloat('aipp_op', 'min_chn_1'))
        tmp_min_params.append(0 if option_list.count('min_chn_2') == 0 else cfg.getfloat('aipp_op', 'min_chn_2'))
        tmp_min_params.append(0 if option_list.count('min_chn_3') == 0 else cfg.getfloat('aipp_op', 'min_chn_3'))

        range_ok = True
        for i in range(0, 4):
            range_ok = range_ok and (0 <= tmp_min_params[i] <= 255)
        if range_ok is False:
            raise RuntimeError('wrong aipp config file content!')

        session.aipp_set_dtc_pixel_min(tmp_min_params)

    @classmethod
    def aipp_set_pixel_var_reci(cls, cfg, session, option_list):
        tmp_reci_params = list()
        tmp_reci_params.append(
            0 if option_list.count('var_reci_chn_0') == 0 else cfg.getfloat('aipp_op', 'var_reci_chn_0')
        )
        tmp_reci_params.append(
            0 if option_list.count('var_reci_chn_1') == 0 else cfg.getfloat('aipp_op', 'var_reci_chn_1')
        )
        tmp_reci_params.append(
            0 if option_list.count('var_reci_chn_2') == 0 else cfg.getfloat('aipp_op', 'var_reci_chn_2')
        )
        tmp_reci_params.append(
            0 if option_list.count('var_reci_chn_3') == 0 else cfg.getfloat('aipp_op', 'var_reci_chn_3')
        )

        range_ok = True
        for i in range(0, 4):
            range_ok = range_ok and (-65504 <= tmp_reci_params[i] <= 65504)
        if range_ok is False:
            raise RuntimeError('wrong aipp config file content!')

        session.aipp_set_dtc_pixel_min(tmp_reci_params)

    # 各种模型
    @classmethod
    def get_without_dymaipp_om_path(cls):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", "pth_resnet50_bs4.om")

    @classmethod
    def get_dymaipp_staticshape_om_path(cls):
        return os.path.join(
            TestCommonClass.get_basepath(), cls.model_name, "model", "pth_resnet50_bs4_dymaipp_stcbatch.om"
        )

    @classmethod
    def get_dymaipp_dymbatch_om_path(cls):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", "pth_resnet50_dymaipp_dymbatch.om")

    @classmethod
    def get_dymaipp_dymwh_om_path(cls):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", "pth_resnet50_dymaipp_dymwh.om")

    @classmethod
    def get_multi_dymaipp_om_path(cls):
        return os.path.join(TestCommonClass.get_basepath(), cls.model_name, "model", "multi_dym_aipp_model.om")

    # 各种输入的aipp具体参数配置文件
    @classmethod
    def get_actual_aipp_config(cls):
        return os.path.join(os.path.dirname(__file__), "../", "aipp_config_files", "actual_aipp_cfg.config")

    @classmethod
    def get_aipp_config_param_overflowed(cls):
        return os.path.join(
            os.path.dirname(__file__), "../", "aipp_config_files", "actual_aipp_cfg_param_overflowed.config"
        )

    @classmethod
    def get_aipp_config_lack_param(cls):
        return os.path.join(os.path.dirname(__file__), "../", "aipp_config_files", "actual_aipp_cfg_lack_param.config")

    @classmethod
    def get_aipp_config_multi_input(cls):
        return os.path.join(os.path.dirname(__file__), "../", "aipp_config_files", "actual_aipp_cfg_multi_input.config")

    @classmethod
    def get_aipp_config_lack_title(cls):
        return os.path.join(os.path.dirname(__file__), "../", "aipp_config_files", "actual_aipp_cfg_lack_title.config")

    def init(self):
        self.model_name = "resnet50"

    # 各种测试场景
    def test_infer_dymaipp_staticshape(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_dymaipp_staticshape_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_staticbatch()
        # only need call this functon compare infer_simple
        self.load_aipp_config_file(session, self.get_actual_aipp_config(), 4)
        session.check_dym_aipp_input_exist()

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name]
        feeds = {session.get_inputs()[0].name: tensor}

        outputs = session.run(outnames, feeds)
        logger.info("outputs:{}".format(outputs))

        for out in outputs:
            out.to_host()
        logger.info(session.sumary())

    def test_infer_dymaipp_dymbatch(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_dymaipp_dymbatch_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_dynamic_batchsize(2)
        # only need call this functon compare infer_simple
        self.load_aipp_config_file(session, self.get_actual_aipp_config(), 4)
        session.check_dym_aipp_input_exist()

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name]
        feeds = {session.get_inputs()[0].name: tensor}

        outputs = session.run(outnames, feeds)
        logger.info("outputs:{}".format(outputs))

        for out in outputs:
            out.to_host()
        logger.info(session.sumary())

    def test_infer_dymaipp_dymwh(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_dymaipp_dymwh_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_dynamic_hw(112, 112)
        # only need call this functon compare infer_simple
        self.load_aipp_config_file(session, self.get_actual_aipp_config(), 1)
        session.check_dym_aipp_input_exist()

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name]
        feeds = {session.get_inputs()[0].name: tensor}

        outputs = session.run(outnames, feeds)
        logger.info("outputs:{}".format(outputs))

        for out in outputs:
            out.to_host()
        logger.info(session.sumary())

    # 模型没有动态aipp input
    def test_infer_no_dymaipp_input(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_without_dymaipp_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_staticbatch()
        # only need call this functon compare infer_simple
        with pytest.raises(RuntimeError) as e:
            session.check_dym_aipp_input_exist()

    # 模型有多个动态aipp input
    def test_infer_multi_dymaipp_input(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_multi_dymaipp_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        # only need call this functon compare infer_simple
        self.load_aipp_config_file(session, self.get_actual_aipp_config(), 1)
        with pytest.raises(RuntimeError) as e:
            session.check_dym_aipp_input_exist()
            logger.info("get --aipp model wrong")

        # create new numpy data according inputs info
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        # convert numpy to pytensors in device
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)

        outnames = [session.get_outputs()[0].name]
        feeds = {session.get_inputs()[0].name: tensor}

        with pytest.raises(RuntimeError) as e:
            outputs = session.run(outnames, feeds)
            logger.info("outputs:{}".format(outputs))

    # --aipp_config 缺少[aipp_op]标识
    def test_infer_aipp_cfg_lack_title(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_dymaipp_staticshape_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_staticbatch()
        # only need call this functon compare infer_simple
        session.check_dym_aipp_input_exist()
        with pytest.raises(RuntimeError) as e:
            self.load_aipp_config_file(session, self.get_aipp_config_lack_title(), 4)
            logger.info("get --aipp_config wrong")

    # --aipp_config 缺少 必备参数
    def test_infer_aipp_cfg_lack_param(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_dymaipp_staticshape_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_staticbatch()
        # only need call this functon compare infer_simple
        session.check_dym_aipp_input_exist()
        with pytest.raises(RuntimeError) as e:
            self.load_aipp_config_file(session, self.get_aipp_config_lack_title(), 4)
            logger.info("get --aipp_config wrong")

    # --aipp_config 参数超出范围限制
    def test_infer_aipp_cfg_param_overflowed(self):
        device_id = 0
        options = aclruntime.session_options()
        model_path = self.get_dymaipp_staticshape_om_path()
        session = aclruntime.InferenceSession(model_path, device_id, options)
        session.set_staticbatch()
        # only need call this functon compare infer_simple
        session.check_dym_aipp_input_exist()
        with pytest.raises(RuntimeError) as e:
            self.load_aipp_config_file(session, self.get_aipp_config_param_overflowed(), 4)
            logger.info("get --aipp_config wrong")
