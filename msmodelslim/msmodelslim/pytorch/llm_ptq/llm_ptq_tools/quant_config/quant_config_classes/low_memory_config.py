#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from ascend_utils.common.security import check_type
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.offloaded_state_dict import OFFLOAD_DISK, OFFLOAD_MEMORY
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param

SUPPORT_OFFLOAD_TYPE = [OFFLOAD_DISK, OFFLOAD_MEMORY]


class LowMemoryConfig(BaseConfig):
    """
    simulate tp的Config
    """

    def __init__(self,
                 last_config: BaseConfig,
                 offload_type: str = OFFLOAD_DISK,
                 should_save_lazily: bool = True):
        super().__init__()
        # 获取上一个config的所有属性，利用上一个config实例完成初始化，并进行参数的修改
        self._init_attribute_by_config(last_config)

        # 更新当前特有的属性
        self.is_adapter_enabled = True # 开启低显存低内存
        self.offload_type = offload_type # state_dict 下放位置
        self.should_save_lazily = should_save_lazily # 是否懒计算权重

        # 校验参数
        check_and_generate_config_param(self)
        if offload_type not in SUPPORT_OFFLOAD_TYPE:
            raise ValueError(f'offload_type should be in {SUPPORT_OFFLOAD_TYPE}, but got {offload_type}.')
        check_type(should_save_lazily, bool, param_name='should_save_lazily')
