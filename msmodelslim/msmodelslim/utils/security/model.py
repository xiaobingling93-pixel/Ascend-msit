# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.security import get_valid_read_path

MAX_KEY_LENGTH = 256
MAX_JSON_LENGTH = 4096


class SafeGenerator:
    def __init__(self):
        pass

    @staticmethod
    def get_config_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=False)
        try:
            config = AutoConfig.from_pretrained(model_path, local_files_only=True, **kwargs)
        except Exception as err:
            raise InvalidModelError('Get config from pretrained failed.',
                                    action=f"Please check config files in the model path. "
                                           f"If the file exists, make sure the folder's owner has execute permission."
                                    ) from err
        return config

    @staticmethod
    def get_model_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=False)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)
        except Exception as err:
            raise InvalidModelError('Get model from pretrained failed.',
                                    action=f"Please check model weights files in the model path. "
                                           f"If the file exists, make sure the folder's owner has execute permission."
                                    ) from err
        return model

    @staticmethod
    def get_tokenizer_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=False)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)
        except Exception as err:
            raise InvalidModelError('Get tokenizer from pretrained failed.',
                                    action=f"Please check tokenizer files in the model path. "
                                           f"If the file exists, make sure the folder's owner has execute permission."
                                    ) from err
        return tokenizer

    @staticmethod
    def load_jsonl(dataset_path, key_name='inputs_pretokenized'):
        dataset = []
        if dataset_path == "humaneval_x.jsonl":
            key_name = 'prompt'
        with os.fdopen(os.open(dataset_path, os.O_RDONLY, 0o600),
                       'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                data = json.loads(line)
                text = data.get(key_name, line)
                dataset.append(text)
        return dataset
