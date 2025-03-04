# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
from dataclasses import dataclass
from ascend_utils.common.security import check_type, get_valid_path, get_valid_write_path


@dataclass
class OmniAttentionConfig:
    pool_size: int = 50
    num_mutation: int = 10
    model_path: str = None
    save_path: str = None
    seed: int = 42
    '''
    Args:
        pool_size: 数据类型为int，遗传算法初始化个体数量，默认值为50
        num_mutation: 数据类型为int，每次进化变异个体数量，默认值为10
        model_path: 数据类型为str，模型路径
        save_path: 数据类型为str，pattern保存路径
        seed: 数据类型为int，随机种子
    '''

    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("Please specify path to HF model and tokenizer.")
        if self.save_path is None:
            raise ValueError("Please verify pattern save path")

        check_type(self.model_path, str, param_name="model_path")
        self.model_path = get_valid_path(self.model_path)
        self._model_name = os.path.basename(os.path.abspath(self.model_path))

        check_type(self.save_path, str, param_name="save_path")
        get_valid_write_path(self.save_path, is_dir=True)

        if not isinstance(self.pool_size, int) or self.pool_size <= 0:
            raise TypeError("pool_size must be positive int")

        if not isinstance(self.num_mutation, int):
            raise TypeError("num_mutation must be int")

        if not isinstance(self.seed, int):
            raise TypeError("random seed must be int")

