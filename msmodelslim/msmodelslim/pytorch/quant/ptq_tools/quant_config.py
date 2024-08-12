# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from ascend_utils.common.security import check_type, check_int, check_element_type

# quant_mode
# 0 : data-free
# 1 : min-max
QUANT_MODE_LIST = [0, 1]

# act_method
# 0 : data-free
# 1 : min-max
# 2 : histogram
ACT_METHOD_LIST = [0, 1, 2]


class QuantConfig:
    """ The configuration for knowledge distillation."""

    def __init__(
        self,
        w_bit=8,
        a_bit=8,
        w_signed=True,
        a_signed=False,
        w_sym=True,
        a_sym=False,
        input_shape=None,
        act_quant=True,
        act_method=0,
        quant_mode=0,
        disable_names=None,
        amp_num=0,
        keep_acc=None,
        sigma=25,
    ):
        # Basic setting
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.w_signed = w_signed
        self.a_signed = a_signed
        self.w_sym = w_sym
        self.a_sym = a_sym
        if input_shape is None:
            input_shape = []
        self.input_shape = input_shape
        self.act_quant = act_quant
        self.sigma = sigma
        # name list for disabled modules
        if disable_names is None:
            disable_names = []
        self.disable_names = disable_names
        # number of layers for AMP fallback
        self.amp_num = amp_num

        # Keep accuracy control, [bool, int] or bool
        # admm is for data-free/label-free, easy_quant is for data-free
        # round_opt is for label-free
        if keep_acc is None:
            self.keep_acc = {'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}
        else:
            self.keep_acc = keep_acc

        # act_method
        # 0 : data-free
        # 1 : min-max
        # 2 : histogram
        self.act_method = act_method
        # quant_mode
        # 0 : data-free
        # 1 : min-max
        self.quant_mode = quant_mode
        self._check_params()

    def _check_params(self):

        if not isinstance(self.w_bit, int) or self.w_bit != 8:
            raise TypeError("w_bit must be 8, please check it.")
        if not isinstance(self.a_bit, int) or self.a_bit != 8:
            raise TypeError("a_bit must be 8, please check it.")

        if 'admm' not in self.keep_acc:
            raise ValueError("admm should be in keep_accuracy")
        if 'easy_quant' not in self.keep_acc:
            raise ValueError("easy_quant should be in keep_accuracy")
        if 'round_opt' not in self.keep_acc:
            raise ValueError("round_opt should be in keep_accuracy")

        check_int(self.amp_num, min_value=0, param_name='amp_num')
        check_int(self.sigma, min_value=0, max_value=100, param_name='sigma')

        if not isinstance(self.disable_names, list):
            raise TypeError("disable_names must be list, please check it.")

        if not isinstance(self.input_shape, list):
            raise TypeError("input_shape must be list, please check it.")
        if len(self.input_shape) != 3 and len(self.input_shape) != 4:
            raise ValueError("input_shape must be 3D (unbatched) or 4D (batched), please check it.")
        check_element_type(self.input_shape, element_type=int, value_type=list, param_name="input_shape")

        if not isinstance(self.act_quant, bool):
            raise ValueError("act_quant is invalid, please check it.")
        if not isinstance(self.w_signed, bool):
            raise ValueError("w_signed is invalid, please check it.")
        if not isinstance(self.a_signed, bool):
            raise ValueError("a_signed is invalid, please check it.")
        if not isinstance(self.w_sym, bool):
            raise ValueError("w_sym is invalid, please check it.")
        if not isinstance(self.a_sym, bool):
            raise ValueError("a_sym is invalid, please check it.")

        if self.act_method not in ACT_METHOD_LIST:
            raise ValueError("act_method is invalid, please check it.")

        if self.quant_mode not in QUANT_MODE_LIST:
            raise ValueError("quant_mode is invalid, please check it.")

