import torch.nn

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.utils import judge_model_with_accelerate, \
    judge_module_with_accelerate

__enabled_adapter = False
__model_compatible = {}


def enable_adapter():
    global __enabled_adapter
    __enabled_adapter = True


def disable_adapter():
    global __enabled_adapter
    __enabled_adapter = False


def enabled_adapter() -> bool:
    return __enabled_adapter


def check_model_compatible(model: torch.nn.Module):
    if not enabled_adapter():
        return False

    if model not in __model_compatible:
        __model_compatible[model] = judge_model_with_accelerate(model)
    return __model_compatible[model]
