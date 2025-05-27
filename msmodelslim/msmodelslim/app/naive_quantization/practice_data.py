# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from dataclasses import dataclass
from typing import List


@dataclass
class Metadata:
    config_id: str
    score: float
    verified_model_types: List[str]  # e.g., ['LLaMa3.1-70B', 'Qwen2.5-72B']
    label: dict  # e.g., {'w_bit': 8, 'a_bit': 8, 'is_sparse': True, 'kv_cache': True}


@dataclass
class QuantizationConfig:
    tokenizer_cfg: dict = None
    model_cfg: dict = None

    # anti
    anti_cfg: dict = None
    anti_params: dict = None

    # calib
    calib_cfg: dict = None
    calib_params: dict = None
    calib_save_params: dict = None

    # quantization parameters
    batch_size: int = 4
    anti_file: str = None
    calib_file: str = None


@dataclass
class CustomizedParams:
    model_path: str = ""
    save_path: str = ""
    device: str = "npu"  # device type
    trust_remote_code: bool = False


@dataclass
class ConfigTask:
    metadata: Metadata
    specific: QuantizationConfig
    customized_config: CustomizedParams = None


def load_specific_config(yaml_spec: dict) -> QuantizationConfig:
    """Load specific configuration from YAML spec"""
    config = QuantizationConfig()

    config.tokenizer_cfg = yaml_spec.get('tokenizer_cfg', {})
    config.model_cfg = yaml_spec.get('model_cfg', {})
    config.anti_cfg = yaml_spec.get('anti_cfg', None)
    config.anti_params = yaml_spec.get('anti_params', {})
    config.calib_cfg = yaml_spec.get('calib_cfg', {})
    config.calib_params = yaml_spec.get('calib_params', {})
    config.calib_save_params = yaml_spec.get('calib_save_params', {})
    config.batch_size = yaml_spec.get('batch_size', 4)
    config.anti_file = yaml_spec.get('anti_file', None)
    config.calib_file = yaml_spec.get('calib_file', '../../../example/common/teacher_qualification.jsonl')
    return config
