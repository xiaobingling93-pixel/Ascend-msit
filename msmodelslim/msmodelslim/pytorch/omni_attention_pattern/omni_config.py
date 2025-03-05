# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
from dataclasses import dataclass
from ascend_utils.common.security import check_type, get_valid_read_path, get_valid_write_path


@dataclass
class OmniAttentionConfig:
    pool_size: int = 50
    num_mutation: int = 10
    model_path: str = None
    save_path: str = None
    seed: int = 42
    '''
    Args:
        pool_size: 数据类型为int，遗传算法初始化个体数量，应大于0，默认值为50
        num_mutation: 数据类型为int，每次进化变异个体数量，应大于0，，默认值为10
        model_path: 数据类型为str，模型路径
        save_path: 数据类型为str，pattern保存路径
        seed: 数据类型为int，随机种子
    '''

    def __post_init__(self):
        if self.model_path is None or len(self.model_path) == 0:
            raise ValueError("Please specify path to HF model and tokenizer, model_path can't be empty.")
        if self.save_path is None or len(self.save_path) == 0:
            raise ValueError("Please verify pattern save path, save_path can't be empty")

        check_type(self.num_mutation, int, param_name="num_mutation")
        check_type(self.seed, int, param_name="seed")
        check_type(self.pool_size, int, param_name="pool_size")

        check_type(self.model_path, str, param_name="model_path")
        self.model_path = get_valid_read_path(self.model_path, is_dir=True)
        self._model_name = os.path.basename(os.path.abspath(self.model_path))

        check_type(self.save_path, str, param_name="save_path")
        get_valid_write_path(self.save_path, is_dir=True)

        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive int")
        if self.num_mutation <= 0:
            raise ValueError("num_mutation must be positive int")

