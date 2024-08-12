# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import torch
import torch.nn as nn
from msmodelslim import logger


def check_torch_module(model):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a Torch.nn.Module instance. Not {}".format(type(model)))


def validate_device(dev_type, dev_id, device_candidates):
    if dev_type not in device_candidates:
        supported_device_types = ', '.join(device_candidates)
        raise ValueError("Device type must be in choices [{}]"
                         .format(supported_device_types))

    if dev_type == "cpu":
        if dev_id is not None:
            logger.warning("`cpu` is set as `dev_type`, `dev_id` cannot be specified manually!")
            dev_id = None
        device = "cpu"
    elif dev_type == "npu":
        try:
            import torch_npu
        except ImportError as e:
            raise ModuleNotFoundError("`torch_npu` cannot be found! Please make sure it correctly installed"
                                      "and can be import without any issues") from e

        if dev_id and not isinstance(dev_id, int):
            raise TypeError("Configuration param `dev_id` cannot be correctly parsed! "
                            "Please make sure `int` is input")
        if dev_id is None:
            default_device = torch.npu.current_device()
            logger.warning("No `dev_id` of npu device is configured, default device id `{}` is set instead."
                           .format(default_device))
            dev_id = default_device
        try:
            torch.npu.get_device_name(dev_id)
        except AssertionError as e:
            raise ValueError("Configuration param `dev_id` cannot be correctly parsed! "
                             "Please make sure a valid device id is input") from e
        device = torch.device("npu:{}".format(dev_id))
    elif dev_type == "gpu":
        device = torch.device("cuda:{}".format(dev_id))
    else:
        device = dev_type

    return device, dev_id
