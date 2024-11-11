import os
import torch
import pytest
from transformers import AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig


def get_fake_llama_model_and_tokenizer():
    """
    获取一个随机的、非常小的Llama模型以及其所对应的tokenizer，用于验证工具中某些数值算法的正确性
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = LlamaConfig.from_json_file(config_path)
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    return LlamaForCausalLM(config), tokenizer


KEY_INPUT_IDS = "input_ids"
KEY_ATTENTION_MASK = "attention_mask"
STR_TEST_PROMPT = "Hello world"
RETURN_TENSOR_TYPE = "pt"


def helper_test_anti_outlier_numeric(anti_method):
    """
    测试各个版本的异常值抑制的数值正确性，相较于异常值抑制之前的模型输出，异常值抑制后的模型输出不应该发生显著改变
    """

    model, tokenizer = get_fake_llama_model_and_tokenizer()

    test_prompt = tokenizer(STR_TEST_PROMPT, return_tensors=RETURN_TENSOR_TYPE, padding=True, truncation=True)

    output_logits_before_anti = model(test_prompt[KEY_INPUT_IDS]).logits

    dataset_calib = [[test_prompt[KEY_INPUT_IDS], test_prompt.data[KEY_ATTENTION_MASK]]]
    anti_config = AntiOutlierConfig(anti_method=anti_method)
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()

    output_logits_after_anti = model(test_prompt[KEY_INPUT_IDS]).logits

    if not torch.allclose(
        output_logits_before_anti, output_logits_after_anti, atol=1e-5
    ):
        pytest.fail()


@pytest.mark.parametrize("anti_method", [pytest.param(anti_method) for anti_method in ["m1", "m2", "m3", "m4", "m5"]])
def test_anti_outlier_numeric(anti_method):
    helper_test_anti_outlier_numeric(anti_method)