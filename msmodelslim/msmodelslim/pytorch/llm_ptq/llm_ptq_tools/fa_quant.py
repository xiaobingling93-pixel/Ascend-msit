# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

import logging
import warnings
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Optional

import torch
from transformers.configuration_utils import PretrainedConfig

from ascend_utils.common.security import check_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import linear_quantization_params
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter import fa_quant_adapter
from msmodelslim import logger as msmodelslim_logger
_SUPPORT_RECALL_WINDOW = False
try:
    from msmodelslim.pytorch.lowbit.atomic_power_outlier import recall_window
except ImportError:
    msmodelslim_logger.warning(
        "The current CANN version does not support recall_window method."
    )
else:
    _SUPPORT_RECALL_WINDOW = True


class TensorType(Enum):
    Q = "q"
    K = "k"
    V = "v"

    @classmethod
    def get_values(cls):
        return [t.value for t in cls]


class QKVQuantizer:
    def __init__(self):
        self.bit = None
        self.sym = None
        self.num_head = None
        self._min_values = None
        self._max_values = None
        self.scale = None
        self.offset = None
        self.is_record = False
        self.single_record = None
        self.ratio = None

    def configure(self, bit: int, sym: bool, num_head: int, ratio: float):
        self.bit = bit
        self.sym = sym
        self.num_head = num_head
        self.ratio = ratio

    def update(self, samples: torch.Tensor, tp_size: int):
        if self.is_record:
            self.single_record = samples
            self.is_record = False

        batch_size, num_head, seq_len, head_dim = samples.shape
        if tp_size == 0:
            raise ZeroDivisionError("tp size can not be zero")
        if not _SUPPORT_RECALL_WINDOW:
            num_head_per_device = self.num_head // tp_size
            samples = samples.contiguous().view(tp_size * num_head_per_device, -1)
            samples_max = samples.max(dim=-1, keepdim=True)[0]
            samples_min = samples.min(dim=-1, keepdim=True)[0]
        else:
            samples = samples.contiguous().view(tp_size * num_head, -1)
            samples_min, samples_max = recall_window(samples, self.ratio, -1, True)

        ## min value
        if self._min_values is None:
            self._min_values = samples_min
        else:
            index = samples_min < self._min_values
            self._min_values[index] = samples_min[index]
        ## max value
        if self._max_values is None:
            self._max_values = samples_max
        else:
            index = samples_max > self._max_values
            self._max_values[index] = samples_max[index]

    def get_scale_offset(
            self,
            states_tensor: torch.Tensor = None,
            tp_size: int = None,
            update: bool = False
    ):
        if update:
            self.update(states_tensor, tp_size)
            self.scale, self.offset = self._calculate_scale_offset()
        return self.scale, self.offset

    def is_calibrated(self):
        return self.scale is not None and self.offset is not None

    def reset_quant_params(self):
        self.scale, self.offset = None, None
    
    def enable_record(self):
        self.is_record = True
    
    def clear_record(self):
        self.single_record = None

    def _calculate_scale_offset(self, integral_zero_point: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self._max_values.dtype
        scale, offset = linear_quantization_params(
            self.bit,
            self._min_values,
            self._max_values,
            sym=self.sym,
            integral_zero_point=integral_zero_point
        )
        if dtype == torch.bfloat16:
            scale = scale.to(dtype)
            offset = offset.to(dtype)
        return scale, offset


class FAQuantizer:
    def __init__(self, config, logger):
        def check_config(config):
            required_attrs = ["num_attention_heads", "hidden_size"]
            for attr in required_attrs:
                if not hasattr(config, attr):
                    raise AttributeError(f"FAQuantizer's config needs attributes: '{attr}'")
            
            check_type(config.num_attention_heads, int, "num_attention_heads in config")
            check_type(config.hidden_size, int, "hidden_size in config")
            
            # 如果不存在num_key_value_heads，则设置为num_attention_heads的值
            if not hasattr(config, "num_key_value_heads"):
                config.num_key_value_heads = config.num_attention_heads
                logger.warning("Failed to obtain `num_key_value_heads`, assuming Multi-head Attention by default.")
            else:
                check_type(config.num_key_value_heads, int, "num_key_value_heads in config")

        check_type(logger, logging.Logger, param_name="logger")
        check_config(config)
        
        self.tp_size = None
        self.num_head = None
        self.num_kv_head = None # GQA模型的KV所用到的注意力头数量，如果是MHA模型则与num_head相同
        self.head_dim = None
        self.is_calib = False
        self.dequant_infer = False
        self.logger = logger
        self.debug_mode = False

        self.q_observer = QKVQuantizer()
        self.k_observer = QKVQuantizer()
        self.v_observer = QKVQuantizer()

        self._set_head_params(config)

        self.processed_types = set()

    def quant(self, states_tensor: torch.Tensor, qkv: TensorType):
        self.processed_types.add(qkv)
        check_type(states_tensor, torch.Tensor, param_name="states_tensor")

        if not self.is_calib and not self.dequant_infer:
            return states_tensor

        if TensorType(qkv) == TensorType.Q:
            scale, offset = self.q_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.q_observer.num_head // self.tp_size
        elif TensorType(qkv) == TensorType.K:
            scale, offset = self.k_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.k_observer.num_head // self.tp_size
        elif TensorType(qkv) == TensorType.V:
            scale, offset = self.v_observer.get_scale_offset(
                states_tensor=states_tensor,
                tp_size=self.tp_size,
                update=not self.dequant_infer
            )
            num_head_per_device = self.v_observer.num_head // self.tp_size
        else:
            values_string = ", ".join(TensorType.get_values())
            raise ValueError(f"Unsupported current TensorType. "
                             f"Please confirm if the parameter is in `{values_string}`")

        expected_types = {"q", "k", "v"}
        if self.processed_types != expected_types:
            missing = expected_types - self.processed_types
            raise RuntimeError(f"Missing qkv types:{missing}. "
                               f"Please ensure all {expected_types} are processed.")

        # only observe the quant params, not quantize and dequantize the tensor
        return states_tensor

    def configure(self, bit: int, sym: bool, tp_size: int = 1):
        self.tp_size = tp_size

        self.q_observer.configure(bit, sym, self.num_head, 0.9999)
        self.k_observer.configure(bit, sym, self.num_kv_head, 0.9999)
        self.v_observer.configure(bit, sym, self.num_kv_head, 1.0)
        
    def set_head_params(self, num_head: int, head_dim: int, num_kv_head: int):
        self.num_head = num_head
        self.head_dim = head_dim
        self.num_kv_head = num_kv_head
    
    def is_calibrated(self):
        return self.q_observer.is_calibrated() and self.k_observer.is_calibrated() and self.v_observer.is_calibrated()

    def disable_calibration(self):
        self.is_calib = False
        if self.is_calibrated():
            self.dequant_infer = True

    def enable_calibration(self):
        self.is_calib = True
        self.q_observer.reset_quant_params()
        self.k_observer.reset_quant_params()
        self.v_observer.reset_quant_params()
    
    def record_once(self):
        self.q_observer.enable_record()
        self.k_observer.enable_record()
        self.v_observer.enable_record()
    
    def reset(self):
        self.q_observer.reset_quant_params()
        self.k_observer.reset_quant_params()
        self.v_observer.reset_quant_params()

    def export_quant_params(self) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        q_scale, q_offset = self.q_observer.get_scale_offset()
        k_scale, k_offset = self.k_observer.get_scale_offset()
        v_scale, v_offset = self.v_observer.get_scale_offset()
        return (
            q_scale.to(device="cpu"),
            k_scale.to(device="cpu"),
            v_scale.to(device="cpu")
        ), (
            q_offset.to(device="cpu", dtype=torch.int8),
            k_offset.to(device="cpu", dtype=torch.int8),
            v_offset.to(device="cpu", dtype=torch.int8)
        )

    def _set_head_params(self, config):  
        num_head = config.num_attention_heads
        hidden_size = config.hidden_size
        
        if num_head == 0:
            raise ValueError("`num_attention_heads` cannot be zero.")
        head_dim = hidden_size // num_head
        
        num_key_value_heads = config.num_key_value_heads
        
        # if GQA（num_key_value_heads != num_head）
        if num_key_value_heads != num_head:
            self.logger.info(
                f"Using Grouped Query Attention (GQA): "
                f"num_key_value_heads={num_key_value_heads}, num_head={num_head}"
            )
        
        self.set_head_params(num_head, head_dim, num_key_value_heads)


def configure_fa(model: torch.nn.Module, bit: int = 8, sym: bool = True, tp_size: int = 1):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.configure(bit, sym, tp_size)


def enable_fa_calibration(model: torch.nn.Module, skip_modules: List[str] = None):
    """
    启用FA校准
    
    Args:
        model: 需要校准的模型
        skip_modules: 需要跳过的模块名称列表
    """
    if skip_modules is None:
        skip_modules = []

    for name, module in model.named_modules():
        if name in skip_modules:
            continue

        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.enable_calibration()


def disable_fa_calibration(model: torch.nn.Module):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.disable_calibration()


def enable_fa_quantizer_record(model: torch.nn.Module):
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            module.fa_quantizer.record_once()


def collect_fa_quantizer_record(model: torch.nn.Module):
    records = {}
    for name, module in model.named_modules():
        if is_attn_module_and_then_check_quantizer(module, name):
            records[f"{name}.query_states"] = module.fa_quantizer.q_observer.single_record
            records[f"{name}.key_states"] = module.fa_quantizer.k_observer.single_record
            records[f"{name}.value_states"] = module.fa_quantizer.v_observer.single_record
            module.fa_quantizer.q_observer.clear_record()
            module.fa_quantizer.k_observer.clear_record()
            module.fa_quantizer.v_observer.clear_record()
    return records
    

def export_fa_quant_params(module: torch.nn.Module, name: str) -> Tuple[dict, ...]:
    quant_params_scale = defaultdict(torch.Tensor)
    quant_params_offset = defaultdict(torch.Tensor)
    attach_map = defaultdict(list)

    param_types = TensorType.get_values()

    if module.fa_quantizer.is_calibrated():
        scales, offsets = module.fa_quantizer.export_quant_params()

    for i, param_type in enumerate(param_types):
        scale_name = f"{name}.fa_{param_type}.scale"
        offset_name = f"{name}.fa_{param_type}.offset"
        
        if module.fa_quantizer.is_calibrated():
            quant_params_scale[scale_name] = scales[i]
            quant_params_offset[offset_name] = offsets[i]

        attach_map[name].extend([scale_name, offset_name])

    return quant_params_scale, quant_params_offset, attach_map


def is_attn_module_and_then_check_quantizer(module: torch.nn.Module, module_name: str) -> bool:
    # flux有Attention类,hunyuan的attn是在MMSingleStreamBlock, MMDoubleStreamBlock中
    if all(x not in module.__class__.__name__ for x in ["Attention", "MMSingleStreamBlock", "MMDoubleStreamBlock"]):
        return False
    if hasattr(module, "fa_quantizer") and isinstance(module.fa_quantizer, FAQuantizer):
        return True
    if hasattr(module, "fa_quantizer") and not isinstance(module.fa_quantizer, FAQuantizer):
        raise AttributeError(f"`FAQuantizer` is not detected in {module_name}. "
                                f"Please check the modeling file and insert FAQuantizer in the correct place.")
    return False


class AttentionType(Enum):
    """注意力机制类型"""
    MHA = "mha"  # Multi-Head Attention
    MQA = "mqa"  # Multi-Query Attention
    GQA = "gqa"  # Group-Query Attention
    MLA = "mla"  # Multi-Head Latent Attention


class ForwardFactory:
    """用于管理不同模型类型和注意力类型/处理器类型的forward函数适配器的工厂类"""

    _forward_adapters = {}

    @classmethod
    def register(cls, model_type: str, attn_or_processor_type: str):
        """装饰器，用于注册forward适配器
        
        Args:
            model_type: 模型/模块类型，如 'deepseekv2', 'llama',
                                    'FluxTransformerBlock','FluxSingleTransformerBlock' 等
            attn_or_processor_type: 注意力类型/处理器类型，如 'mha', 'mqa', 'gqa', 'mla', 
                                    'FluxAttnProcessor2_0', 'FluxSingleAttnProcessor2_0'等
        """

        def decorator(func):
            key = (model_type, attn_or_processor_type)
            cls._forward_adapters[key] = func
            return func

        return decorator

    @classmethod
    def get_forward_adapter(cls, model_type: str, attn_or_processor_type: str):
        """获取指定模型类型和注意力类型的forward适配器"""
        key = (model_type, attn_or_processor_type)
        if key not in cls._forward_adapters:
            raise ValueError(
                f"Unsupported combination: model_type={model_type}, attn_or_processor_type={attn_or_processor_type}"
                )
        
        return cls._forward_adapters[key]

    @classmethod
    def detect_attention_type(cls, module: torch.nn.Module) -> str:
        """检测模块的注意力类型
        
        Args:
            module: 注意力模块
            
        Returns:
            str: 注意力类型
        """
        if not hasattr(module, "num_key_value_heads"):
            return AttentionType.MHA.value

        if module.num_key_value_heads == module.num_attention_heads:
            return AttentionType.MHA.value
        elif module.num_key_value_heads == 1:
            return AttentionType.MQA.value
        elif module.num_key_value_heads < module.num_attention_heads:
            return AttentionType.GQA.value

        return AttentionType.MHA.value


class ModelType(Enum):
    FLUX = "flux"
    HYVIDEO = "hyvideo"
    DEFAULT = "default"


class ModelTypeDetector:
    """模型类型检测器"""
    @staticmethod
    def detect_model_type(model: torch.nn.Module) -> 'ModelType':
        first_module_name, first_module = list(model.named_modules())[0][1]
        module_class_name = first_module.__class__.__name__
        
        if "Flux" in module_class_name:
            return ModelType.FLUX
        elif "HYVideo" in module_class_name:
            return ModelType.HYVIDEO
        else:
            return ModelType.DEFAULT


class ModelConfigFactory:
    @staticmethod
    def get_model_config(model: torch.nn.Module, config: PretrainedConfig, logger: logging.Logger):
        """根据模型类型生成对应的量化配置"""
        model_type = ModelTypeDetector.detect_model_type(model)
        
        if model_type == ModelType.FLUX:
            return ModelConfigFactory._get_flux_config(config, logger)
        elif model_type == ModelType.HYVIDEO:
            return ModelConfigFactory._get_hyvideo_config(config, logger)
        else:
            return config

    @staticmethod
    def _get_flux_config(config: PretrainedConfig, logger: logging.Logger):
        from types import SimpleNamespace
        import torch.distributed as dist

        sp_size = 1 if not config.is_tp else dist.get_world_size()

        config_dict = {
            'num_attention_heads': config.num_attention_heads // sp_size, 
            'hidden_size': config.attention_head_dim * config.num_attention_heads,
            'num_key_value_heads': config.num_attention_heads // sp_size,
        }
        return SimpleNamespace(**config_dict)

    @staticmethod
    def _get_hyvideo_config(config: PretrainedConfig, logger: logging.Logger):
        from types import SimpleNamespace
        import torch.distributed as dist
        
        if dist.is_initialized():
            sp_size = dist.get_world_size()
            logger.info(f"sp_size: {sp_size}")
        else:
            logger.info("sp_size = 1 (not in distributed environment)")
            sp_size = 1

        config_dict = {
            'num_attention_heads': config.heads_num // sp_size, 
            'hidden_size': config.hidden_size // sp_size,
            'num_key_value_heads': config.heads_num // sp_size,
        }
        return SimpleNamespace(**config_dict)


def _install_forward_adapter(
    module: torch.nn.Module,
    module_type: str,
    adapter_type: str,
    module_name: str,
    logger: logging.Logger,
):
    """安装forward适配器
    
    Args:
        module: 要安装适配器的模块
        module_type: 模块类型(用于判断模型类型)
        adapter_type: 适配器类型
        module_name: 模块名称(用于日志记录)
        logger: 日志记录器
    """
    try:
        if not adapter_type:
            return
            
        forward_adapter = ForwardFactory.get_forward_adapter(module_type, adapter_type)
        
        # Flux模型处理逻辑
        if "Flux" in module_type:
            # Flux模型处理processor的情况
            processor_cls = module.processor.__class__
            original_call = module.processor.__call__
            new_call = forward_adapter(original_call).__get__(
                module.processor, processor_cls
            )
            processor_cls.__call__ = new_call
            
        # HYVideo模型处理逻辑
        elif "MMDoubleStreamBlock" in module_type or "MMSingleStreamBlock" in module_type:
            # HYVideo模型处理forward方法的情况
            original_method = getattr(module, adapter_type)
            adapted_method = forward_adapter(original_method).__get__(module, module.__class__)
            setattr(module, adapter_type, adapted_method)
            
        # 默认模型处理逻辑
        else:
            # 默认处理forward方法的情况
            if hasattr(module, "forward"):
                module.forward = forward_adapter(module.forward).__get__(module, module.__class__)
            else:
                raise ValueError(f"Module {module_name} has no forward method to adapt")

        logger.info(f"Successfully installed FAQuantizer for module {module_name}")
        
    except Exception as e:
        logger.error(f"Failed to install FAQuantizer for module {module_name}: {str(e)}")
        raise


def _install_for_flux_model(
    model: torch.nn.Module,
    config: PretrainedConfig,
    logger: logging.Logger,
    skip_layers: List[str],
):
    """为Flux模型安装量化器"""
    processor_map = {
        "FluxTransformerBlock": "FluxAttnProcessor2_0",
        "FluxSingleTransformerBlock": "FluxSingleAttnProcessor2_0"
    }

    for name, module in model.named_modules():
        module_type = module.__class__.__name__
    
        if module_type not in processor_map:
            continue
            
        attn_module = module.attn
        attn_full_name = f"{name}.attn"
        
        if any(skip_name in attn_full_name for skip_name in skip_layers):
            logger.info(f"Skipping {attn_full_name}")
            continue

        if hasattr(attn_module, "fa_quantizer"):
            logger.warning(f"Module {attn_full_name} already has FAQuantizer installed.")
            continue
        
        flux_config = ModelConfigFactory.get_model_config(model, config, logger)
        attn_module.fa_quantizer = FAQuantizer(flux_config, logger=logger)
        logger.info(f"Installed quantizer at {attn_full_name}")

        processor_type = processor_map.get(module_type, None)
        if processor_type:
            _install_forward_adapter(attn_module, module_type, processor_type, attn_full_name, logger)


def _install_for_hyvideo_model(
    model: torch.nn.Module,
    config: PretrainedConfig,
    logger: logging.Logger,
    skip_layers: List[str],
):
    """为HYVideo模型安装量化器"""
    module_map = {
        "MMDoubleStreamBlock": "double_forward",
        "MMSingleStreamBlock": "single_forward"
    }

    for name, module in model.named_modules():
        module_type = module.__class__.__name__
    
        if module_type not in module_map:
            continue
            
        if any(skip_name in name for skip_name in skip_layers):
            logger.info(f"Skipping FAQuantizer installation for module {name}")
            continue

        if hasattr(module, "fa_quantizer"):
            logger.warning(f"Module {name} already has FAQuantizer installed.")
            continue

        hunyuan_config = ModelConfigFactory.get_model_config(model, config, logger)
        module.fa_quantizer = FAQuantizer(hunyuan_config, logger=logger)
        logger.info(f"Installed quantizer at {name}")

        forward_type = module_map.get(module_type, None)
        if forward_type:
            _install_forward_adapter(module, module_type, forward_type, name, logger)


def _install_for_default_model(
    model: torch.nn.Module,
    config: PretrainedConfig,
    logger: logging.Logger,
    skip_layers: List[str],
):
    """为默认模型安装量化器"""
    for name, module in model.named_modules():
        if "Attention" in module.__class__.__name__:
            if any(skip_name in name for skip_name in skip_layers):
                logger.info(f"Skipping FAQuantizer installation for module {name}")
                continue

            if hasattr(module, "fa_quantizer"):
                logger.warning(f"Module {name} already has FAQuantizer installed.")
                continue
            
            module.fa_quantizer = FAQuantizer(config, logger=logger)

            default_attn_type = {
                "deepseekv2": AttentionType.MLA,
                "deepseek_v2": AttentionType.MLA,
                "deepseekv3": AttentionType.MLA,
                "deepseek_v3": AttentionType.MLA,
            }

            attn_type = default_attn_type.get(config.model_type, AttentionType.MHA).value
            if attn_type:
                _install_forward_adapter(module, config.model_type, attn_type, name, logger)


def install_fa_quantizer(
    model: torch.nn.Module,
    config: PretrainedConfig,
    logger: logging.Logger,
    skip_layers: Optional[List[str]] = None,
):
    """为模型安装FAQuantizer
    
    Args:
        model: 需要安装FAQuantizer的模型
        config: 模型配置
        logger: 日志记录器
        skip_layers: 需要跳过的层名列表
    """
    skip_layers = skip_layers or []
    model_type = ModelTypeDetector.detect_model_type(model)

    if model_type == ModelType.FLUX:
        _install_for_flux_model(model, config, logger, skip_layers)
    elif model_type == ModelType.HYVIDEO:
        _install_for_hyvideo_model(model, config, logger, skip_layers)
    else:
        _install_for_default_model(model, config, logger, skip_layers)


# # 全局变量用于保存第一次获取的函数
# FLUX_APPLY_ROTARY_EMB_MINDSPEED = None
# FLUX_APPLY_FA = None
# FLUX_ATTENTION = None

# # 注册 Flux 模型的适配器
# @ForwardFactory.register("FluxTransformerBlock", "FluxAttnProcessor2_0")
# def flux_attn_processor_adapter(original_call):
#     """FluxAttnProcessor2_0 的量化适配器"""
#     global FLUX_APPLY_ROTARY_EMB_MINDSPEED, FLUX_APPLY_FA, FLUX_ATTENTION

#     from importlib import import_module
#     import torch.distributed as dist

#     # 如果已经初始化过，直接使用保存的函数
#     if all([FLUX_APPLY_ROTARY_EMB_MINDSPEED, FLUX_APPLY_FA, FLUX_ATTENTION]):
#         apply_rotary_emb_mindspeed = FLUX_APPLY_ROTARY_EMB_MINDSPEED
#         apply_fa = FLUX_APPLY_FA
#         Attention = FLUX_ATTENTION
#     else:
#         # 第一次调用，从模块中获取并保存
#         flux_attn_processor_module = import_module(original_call.__module__)
        
#         apply_rotary_emb_mindspeed = flux_attn_processor_module.apply_rotary_emb_mindspeed
#         apply_fa = flux_attn_processor_module.apply_fa
#         Attention = flux_attn_processor_module.Attention
        
#         # 保存到全局变量
#         FLUX_APPLY_ROTARY_EMB_MINDSPEED = apply_rotary_emb_mindspeed
#         FLUX_APPLY_FA = apply_fa
#         FLUX_ATTENTION = Attention
    
#     def new_call(
#         self,
#         attn: Attention,
#         hidden_states: torch.FloatTensor,
#         encoder_hidden_states: torch.FloatTensor = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         image_rotary_emb: Optional[torch.Tensor] = None,
#     ) -> torch.FloatTensor:
        
#         input_ndim = hidden_states.ndim
#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#         context_input_ndim = encoder_hidden_states.ndim
#         if context_input_ndim == 4:
#             batch_size, channel, height, width = encoder_hidden_states.shape
#             encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size = encoder_hidden_states.shape[0]

#         # `sample` projections.
#         query = attn.to_q(hidden_states)
#         key = attn.to_k(hidden_states)
#         value = attn.to_v(hidden_states)

#         inner_dim = key.shape[-1]
#         if attn.is_tp:
#             attn_heads = attn.heads // attn.world_size
#         else:
#             attn_heads = attn.heads
#         head_dim = inner_dim // attn_heads

#         query = query.view(batch_size, -1, attn_heads, head_dim)
#         key = key.view(batch_size, -1, attn_heads, head_dim)
#         value = value.view(batch_size, -1, attn_heads, head_dim)

#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)

#         # `context` projections.
#         encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
#         encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
#         encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

#         encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
#             batch_size, -1, attn_heads, head_dim
#         )
#         encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
#             batch_size, -1, attn_heads, head_dim
#         )
#         encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
#             batch_size, -1, attn_heads, head_dim
#         )

#         if attn.norm_added_q is not None:
#             encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
#         if attn.norm_added_k is not None:
#             encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

#         # attention
#         query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
#         key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
#         value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

#         if image_rotary_emb is not None:
#             query = apply_rotary_emb_mindspeed(query, image_rotary_emb)
#             key = apply_rotary_emb_mindspeed(key, image_rotary_emb)

#         # --------------------fa3-----------------------------
#         query = attn.fa_quantizer.quant(query, qkv="q")
#         key = attn.fa_quantizer.quant(key, qkv="k")
#         value = attn.fa_quantizer.quant(value, qkv="v")
#         # --------------------fa3-----------------------------
        
#         hidden_states = apply_fa(query, key, value, attention_mask)
#         hidden_states = hidden_states.to(query.dtype)

#         encoder_hidden_states, hidden_states = (
#             hidden_states[:, : encoder_hidden_states.shape[1]],
#             hidden_states[:, encoder_hidden_states.shape[1]:],
#         )

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)
#         if attn.is_tp:
#             dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)
#         encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
#         if attn.is_tp:
#             dist.all_reduce(encoder_hidden_states, op=dist.ReduceOp.SUM)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
#         if context_input_ndim == 4:
#             encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         return hidden_states, encoder_hidden_states
#     return new_call


# # 全局变量用于保存第一次获取的函数
# FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED = None
# FLUX_SINGLE_APPLY_FA = None
# FLUX_SINGLE_ATTENTION = None


# # 注册 Flux 模型的适配器
# @ForwardFactory.register("FluxSingleTransformerBlock", "FluxSingleAttnProcessor2_0")
# def flux_attn_single_processor_adapter(original_call):
#     """FluxSingleAttnProcessor2_0 的量化适配器"""
#     global FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED, FLUX_SINGLE_APPLY_FA, FLUX_SINGLE_ATTENTION

#     from importlib import import_module
#     import torch.distributed as dist

#     # 如果已经初始化过，直接使用保存的函数
#     if all([FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED, FLUX_SINGLE_APPLY_FA, FLUX_SINGLE_ATTENTION]):
#         apply_rotary_emb_mindspeed = FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED
#         apply_fa = FLUX_SINGLE_APPLY_FA
#         Attention = FLUX_SINGLE_ATTENTION
#     else:
#         # 第一次调用，从模块中获取并保存
#         flux_attn_processor_module = import_module(original_call.__module__)
        
#         apply_rotary_emb_mindspeed = flux_attn_processor_module.apply_rotary_emb_mindspeed
#         apply_fa = flux_attn_processor_module.apply_fa
#         Attention = flux_attn_processor_module.Attention
        
#         # 保存到全局变量
#         FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED = apply_rotary_emb_mindspeed
#         FLUX_SINGLE_APPLY_FA = apply_fa
#         FLUX_SINGLE_ATTENTION = Attention


#     def new_call(
#         self,
#         attn: Attention,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         image_rotary_emb: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

#         query = attn.to_q(hidden_states)
#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states

#         key = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)

#         inner_dim = key.shape[-1]
#         if attn.is_tp:
#             attn_heads = attn.heads // attn.world_size
#         else:
#             attn_heads = attn.heads
#         head_dim = inner_dim // attn_heads

#         query = query.view(batch_size, -1, attn_heads, head_dim)

#         key = key.view(batch_size, -1, attn_heads, head_dim)
#         value = value.view(batch_size, -1, attn_heads, head_dim)

#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)

#         # Apply RoPE if needed
#         if image_rotary_emb is not None:
#             query = apply_rotary_emb_mindspeed(query, image_rotary_emb)
#             key = apply_rotary_emb_mindspeed(key, image_rotary_emb)

#         # --------------------fa3-----------------------------
#         query = attn.fa_quantizer.quant(query, qkv="q")
#         key = attn.fa_quantizer.quant(key, qkv="k")
#         value = attn.fa_quantizer.quant(value, qkv="v")
#         # --------------------fa3-----------------------------

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         hidden_states = apply_fa(query, key, value, attention_mask)
#         hidden_states = hidden_states.to(query.dtype)
#         B, S, H = hidden_states.shape
#         if attn.is_tp:
#             hidden_states_full = torch.empty(
#                 [attn.world_size, B, S, H], dtype=hidden_states.dtype, device=hidden_states.device
#                 )
#             dist.all_gather_into_tensor(hidden_states_full, hidden_states)
#             hidden_states = hidden_states_full.permute(1, 2, 0, 3).reshape([B, S, 2 * H])

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         return hidden_states
#     return new_call


# # 注册 HYVideo 模型的适配器
# @ForwardFactory.register("MMDoubleStreamBlock", "double_forward")
# def hyvideo_mm_double_stream_block_double_forward_adapter(original_forward):
#     """HYVideo 模型的double_forward适配器"""
#     from importlib import import_module
        
#     hyvideo_double_module = import_module(original_forward.__module__)
#     modulate = hyvideo_double_module.modulate
#     rearrange = hyvideo_double_module.rearrange
#     apply_rotary_emb = hyvideo_double_module.apply_rotary_emb
#     attention = hyvideo_double_module.attention
#     parallel_attention = hyvideo_double_module.parallel_attention

#     def new_double_forward(
#             self,
#             img, txt,
#             img_mod1_shift,
#             img_mod1_scale,
#             txt_mod1_shift,
#             txt_mod1_scale,
#             freqs_cis,
#             cu_seqlens_q,
#             cu_seqlens_kv,
#             max_seqlen_q,
#             max_seqlen_kv
#         ):
#         # Prepare image for attention.
#         img_modulated = self.img_norm1(img)
#         img_modulated = modulate(
#             img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
#         )
#         img_qkv = self.img_attn_qkv(img_modulated)
#         img_q, img_k, img_v = rearrange(
#             img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
#         )
#         # Apply QK-Norm if needed
#         img_q = self.img_attn_q_norm(img_q).to(img_v)
#         img_k = self.img_attn_k_norm(img_k).to(img_v)

#         # Apply RoPE if needed.
#         if freqs_cis is not None:
#             img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
#             assert (
#                 img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
#             ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
#             img_q, img_k = img_qq, img_kk

#         # Prepare txt for attention.
#         txt_modulated = self.txt_norm1(txt)
#         txt_modulated = modulate(
#             txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
#         )
#         txt_qkv = self.txt_attn_qkv(txt_modulated)
#         txt_q, txt_k, txt_v = rearrange(
#             txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
#         )
#         # Apply QK-Norm if needed.
#         txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
#         txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

#         # Run actual attention.
#         q = torch.cat((img_q, txt_q), dim=1)
#         k = torch.cat((img_k, txt_k), dim=1)
#         v = torch.cat((img_v, txt_v), dim=1)
#         assert (
#             cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
#         ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        
#         # --------------------fa3-----------------------------
#         q = self.fa_quantizer.quant(q, qkv="q")
#         k = self.fa_quantizer.quant(k, qkv="k")
#         v = self.fa_quantizer.quant(v, qkv="v")
#         # --------------------fa3-----------------------------

#         # attention computation start
#         if not self.hybrid_seq_parallel_attn:
#             attn = attention(
#                 q,
#                 k,
#                 v,
#                 mode="torch",
#                 cu_seqlens_q=cu_seqlens_q,
#                 cu_seqlens_kv=cu_seqlens_kv,
#                 max_seqlen_q=max_seqlen_q,
#                 max_seqlen_kv=max_seqlen_kv,
#                 batch_size=img_k.shape[0],
#             )
#         else:
#             attn = parallel_attention(
#                 self.hybrid_seq_parallel_attn,
#                 q,
#                 k,
#                 v,
#                 img_q_len=img_q.shape[1],
#                 img_kv_len=img_k.shape[1],
#                 cu_seqlens_q=cu_seqlens_q,
#                 cu_seqlens_kv=cu_seqlens_kv,
#                 scale=self.scale
#             )
        
#         return attn
#     return new_double_forward


# @ForwardFactory.register("MMSingleStreamBlock", "single_forward")
# def hyvideo_mm_single_stream_block_single_forward_adapter(original_forward):
#     """HYVideo 模型的 single_forward 适配器"""
#     from importlib import import_module
    
#     hyvideo_single_module = import_module(original_forward.__module__)
#     rearrange = hyvideo_single_module.rearrange
#     apply_rotary_emb = hyvideo_single_module.apply_rotary_emb
#     attention = hyvideo_single_module.attention
#     parallel_attention = hyvideo_single_module.parallel_attention

#     def new_single_forward(
#             self,
#             qkv,
#             freqs_cis,
#             txt_len,
#             x,
#             cu_seqlens_q,
#             cu_seqlens_kv,
#             max_seqlen_q,
#             max_seqlen_kv
#         ):
#         q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

#         # Apply QK-Norm if needed.
#         q = self.q_norm(q).to(v)
#         k = self.k_norm(k).to(v)

#         # Apply RoPE if needed.
#         if freqs_cis is not None:
#             img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
#             img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
#             img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
#             assert (
#                 img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
#             ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
#             img_q, img_k = img_qq, img_kk
#             q = torch.cat((img_q, txt_q), dim=1)
#             k = torch.cat((img_k, txt_k), dim=1)

#         # Compute attention.
#         assert (
#             cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
#         ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        
#         # --------------------fa3-----------------------------
#         q = self.fa_quantizer.quant(q, qkv="q")
#         k = self.fa_quantizer.quant(k, qkv="k")
#         v = self.fa_quantizer.quant(v, qkv="v")
#         # --------------------fa3----------------------------- 

#         # attention computation start
#         if not self.hybrid_seq_parallel_attn:
#             attn = attention(
#                 q,
#                 k,
#                 v,
#                 mode="torch",
#                 cu_seqlens_q=cu_seqlens_q,
#                 cu_seqlens_kv=cu_seqlens_kv,
#                 max_seqlen_q=max_seqlen_q,
#                 max_seqlen_kv=max_seqlen_kv,
#                 batch_size=x.shape[0],
#             )
#         else:
#             attn = parallel_attention(
#                 self.hybrid_seq_parallel_attn,
#                 q,
#                 k,
#                 v,
#                 img_q_len=img_q.shape[1],
#                 img_kv_len=img_k.shape[1],
#                 cu_seqlens_q=cu_seqlens_q,
#                 cu_seqlens_kv=cu_seqlens_kv,
#                 scale=self.scale
#             )
#         # attention computation end
#         return attn       
#     return new_single_forward

# @ForwardFactory.register("deepseekv3", "mla")
# @ForwardFactory.register("deepseek_v3", "mla")
# @ForwardFactory.register("deepseekv2", "mla")
# @ForwardFactory.register("deepseek_v2", "mla")
# def deepseekv2_mla_forward_adapter(original_forward):
#     """DeepSeek V2/V3模型的MLA forward适配器"""

#     from importlib import import_module
#     from transformers import Cache
#     from torch import nn
#     from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import PrepareWeight

#     deepseek_module = import_module(original_forward.__module__)
#     apply_rotary_pos_emb = deepseek_module.apply_rotary_pos_emb

#     def new_forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_value: Optional[Cache] = None,
#             output_attentions: bool = False,
#             use_cache: bool = False,
#             **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if "padding_mask" in kwargs:
#             warnings.warn(
#                 "Passing `padding_mask` is deprecated and will be removed in v4.37.\n"
#                 "Please make sure to use `attention_mask` instead."
#             )
#         bsz, q_len, _ = hidden_states.size()

#         if self.q_lora_rank is None:
#             q = self.q_proj(hidden_states)
#         else:
#             q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
#         q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
#         q_nope, q_pe = torch.split(
#             q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
#         )

#         compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
#         compressed_kv, k_pe = torch.split(
#             compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
#         )
#         compressed_kv = self.kv_a_layernorm(compressed_kv)
#         k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
#         kv_seq_len = k_pe.shape[-2]

#         if past_key_value is not None:
#             if self.layer_idx is None:
#                 raise ValueError(
#                     f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
#                     "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
#                     "with a layer index."
#                 )
            
#             kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

#         cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
#         q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

#         if past_key_value is not None:
#             cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
#             compressed_kv = compressed_kv.unsqueeze(1)
#             k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
#             compressed_kv = compressed_kv.squeeze(1)

#         with PrepareWeight(self.kv_b_proj):
#             kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)

#         q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
#         out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

#         q_nope = torch.matmul(q_nope, q_absorb)

#         # ----------FA3-------------
#         q_nope = self.fa_quantizer.quant(q_nope, qkv="q")
#         compressed_kv = self.fa_quantizer.quant(compressed_kv.unsqueeze(1), qkv="k").squeeze(1)
#         _ = self.fa_quantizer.quant(compressed_kv.unsqueeze(1), qkv="v").squeeze(1)
#         # ----------FA3-------------

#         attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT))
#         attn_weights = attn_weights * self.softmax_scale

#         if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                 f" {attn_weights.size()}"
#             )
#         if attention_mask is None:
#             raise ValueError("Attention mask cannot be None")
#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights + attention_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(
#             attn_weights, dim=-1, dtype=torch.float32
#         ).to(q_pe.dtype)
#         attn_weights = nn.functional.dropout(
#             attn_weights, p=self.attention_dropout, training=self.training
#         )
#         attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
#         attn_output = torch.matmul(attn_output, out_absorb.mT)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value

#     return new_forward


# def install_fa_quantizer(
#         model: torch.nn.Module,
#         config: PretrainedConfig,
#         logger: logging.Logger,
#         skip_layers: Optional[List[str]] = None,
# ):
#     """为模型安装FAQuantizer
    
#     Args:
#         model: 需要安装FAQuantizer的模型
#         config: 模型配置
#         logger: 日志记录器
#         skip_layers: 需要跳过的层名列表
#     """

#     skip_layers = skip_layers or []

#     if "Flux" in list(model.named_modules())[0][1].__class__.__name__ :

#         # 处理器类型映射
#         processor_map = {
#             "FluxTransformerBlock": "FluxAttnProcessor2_0",
#             "FluxSingleTransformerBlock": "FluxSingleAttnProcessor2_0"
#         }

#         for name, module in model.named_modules():
#             module_type = module.__class__.__name__
        
#             # 只处理目标块类型
#             if module_type not in processor_map:
#                 continue
                
#             # 提取块内的Attention模块
#             attn_module = module.attn
#             attn_full_name = f"{name}.attn" 
            
#             if any(skip_name in attn_full_name for skip_name in skip_layers):
#                 logger.info(f"Skipping {attn_full_name}")
#                 continue

#             if hasattr(attn_module, "fa_quantizer"):
#                 logger.warning(f"Module {attn_full_name} already has FAQuantizer installed.")
#                 continue
            
#             # --------------------fa3-----------------------------
#             # from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
#             # from msmodelslim import logger 

#             from types import SimpleNamespace
#             import torch.distributed as dist

#             sp_size = 1 if not config.is_tp else dist.get_world_size()

#             config_dict = {
#                 'num_attention_heads': config.num_attention_heads // sp_size, 
#                 'hidden_size': config.attention_head_dim * config.num_attention_heads,
#                 'num_key_value_heads': config.num_attention_heads // sp_size,
#                 }

#             flux_config = SimpleNamespace(**config_dict)
#             attn_module.fa_quantizer = FAQuantizer(flux_config, logger=logger)
#             logger.info(f"Installed quantizer at {attn_full_name}")
#             # --------------------fa3-----------------------------

#             processor_type = processor_map.get(module_type, None)

#             try:
#                 if processor_type:
#                     forward_adapter = ForwardFactory.get_forward_adapter(module_type, processor_type)
                    
#                     processor_cls = attn_module.processor.__class__

#                     # 获取原始的 __call__ 方法并适配
#                     original_call = attn_module.processor.__call__
#                     new_call = forward_adapter(original_call).__get__(
#                         attn_module.processor, processor_cls
#                     )

#                     # 替换 processor_cls 的 __call__ 方法
#                     processor_cls.__call__ = new_call

#                     logger.info(f"Successfully installed FAQuantizer for module {attn_full_name}")
#             except ValueError as e:
#                 logger.error(f"Failed to install FAQuantizer for module {attn_full_name}: {str(e)}")
#                 raise

#     elif "HYVideo" in list(model.named_modules())[0][1].__class__.__name__:

#         module_map = {
#             "MMDoubleStreamBlock": "double_forward",
#             "MMSingleStreamBlock": "single_forward"
#         }

#         for name, module in model.named_modules():
#             module_type = module.__class__.__name__
        
#             if module_type not in module_map:
#                 continue
                
#             if any(skip_name in name for skip_name in skip_layers):
#                 logger.info(f"Skipping FAQuantizer installation for module {name}")
#                 continue

#             if hasattr(module, "fa_quantizer"):
#                 logger.warning(f"Module {name} already has FAQuantizer installed.")
#                 continue

#             # --------------------fa3-----------------------------
#             from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
#             from msmodelslim import logger 
#             from types import SimpleNamespace
#             import torch.distributed as dist
            
#             if dist.is_initialized():
#                 sp_size = dist.get_world_size()
#                 logger.info(f"sp_size: {sp_size}")
#             else:
#                 logger.info("sp_size = 1 (not in distributed environment)")
#                 sp_size = 1

#             config_dict = {
#                 'num_attention_heads': config.heads_num // sp_size, 
#                 'hidden_size': config.hidden_size // sp_size,
#                 'num_key_value_heads': config.heads_num // sp_size,
#                 }

#             hunyuan_config = SimpleNamespace(**config_dict)
#             module.fa_quantizer = FAQuantizer(hunyuan_config, logger=logger)
#             logger.info(f"Installed quantizer at {name}")
#             # --------------------fa3-----------------------------

#             forward_type = module_map.get(module_type, None)

#             try:
#                 if forward_type:
#                     forward_adapter = ForwardFactory.get_forward_adapter(module_type, forward_type)
        
#                     original_method = getattr(module, forward_type)
#                     adapted_method = forward_adapter(original_method).__get__(module, module.__class__)
#                     setattr(module, forward_type, adapted_method)

#                     logger.info(f"Successfully installed FAQuantizer for module {name}")
#             except ValueError as e:
#                 logger.error(f"Failed to install FAQuantizer for module {name}: {str(e)}")
#                 raise

#     else:
#         for name, module in model.named_modules():
#             if "Attention" in module.__class__.__name__:
#                 if any(skip_name in name for skip_name in skip_layers):
#                     logger.info(f"Skipping FAQuantizer installation for module {name}")
#                     continue

#                 if hasattr(module, "fa_quantizer"):
#                     logger.warning(f"Module {name} already has FAQuantizer installed.")
#                     continue
                
#                 module.fa_quantizer = FAQuantizer(config, logger=logger)

#                 default_attn_type = {
#                     "deepseekv2": AttentionType.MLA,
#                     "deepseek_v2": AttentionType.MLA,
#                     "deepseekv3": AttentionType.MLA,
#                     "deepseek_v3": AttentionType.MLA,
#                 }

#                 attn_type = default_attn_type.get(config.model_type, AttentionType.MHA).value

#                 try:
#                     forward_adapter = ForwardFactory.get_forward_adapter(config.model_type, attn_type)
#                     module.forward = forward_adapter(module.forward).__get__(module, module.__class__)
#                     logger.info(f"Successfully installed FAQuantizer for module {name} with attention type {attn_type}")
#                 except ValueError as e:
#                     logger.error(f"Failed to install FAQuantizer for module {name}: {str(e)}")
#                     raise
