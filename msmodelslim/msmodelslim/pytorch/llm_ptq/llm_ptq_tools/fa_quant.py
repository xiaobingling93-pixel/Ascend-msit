# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging
import warnings
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Optional

import torch
from transformers.configuration_utils import PretrainedConfig

from ascend_utils.common.security import check_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import linear_quantization_params
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

        check_config(config)
            
        check_type(logger, logging.Logger, param_name="logger")

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

        expected_types = {"q", "k", "v"}
        if self.processed_types != expected_types:
            missing = expected_types - self.processed_types
            raise RuntimeError(f"Missing qkv types:{missing}. "
                               f"Please ensure all {expected_types} are processed.")

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
    """用于管理不同模型类型和注意力类型的forward函数适配器的工厂类"""

    _forward_adapters = {}

    @classmethod
    def register(cls, model_type: str, attn_type: str):
        """装饰器，用于注册forward适配器
        
        Args:
            model_type: 模型类型，如 'deepseekv2', 'llama' 等
            attn_type: 注意力类型，如 'mha', 'mqa', 'gqa', 'mla' 等
        """

        def decorator(func):
            key = (model_type, attn_type)
            cls._forward_adapters[key] = func
            return func

        return decorator

    @classmethod
    def get_forward_adapter(cls, model_type: str, attn_type: str):
        """获取指定模型类型和注意力类型的forward适配器"""
        key = (model_type, attn_type)
        if key not in cls._forward_adapters:
            raise ValueError(f"Unsupported combination: model_type={model_type}, attn_type={attn_type}")
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


@ForwardFactory.register("deepseekv3", "mla")
@ForwardFactory.register("deepseek_v3", "mla")
@ForwardFactory.register("deepseekv2", "mla")
@ForwardFactory.register("deepseek_v2", "mla")
def deepseekv2_mla_forward_adapter(original_forward):
    """DeepSeek V2/V3模型的MLA forward适配器"""

    from importlib import import_module
    from transformers import Cache
    from torch import nn
    from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import PrepareWeight

    deepseek_module = import_module(original_forward.__module__)
    apply_rotary_pos_emb = deepseek_module.apply_rotary_pos_emb

    def new_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37.\n"
                "Please make sure to use `attention_mask` instead."
            )
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv_seq_len = k_pe.shape[-2]

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            compressed_kv = compressed_kv.unsqueeze(1)
            k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
            compressed_kv = compressed_kv.squeeze(1)

        with PrepareWeight(self.kv_b_proj):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)

        q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
        out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

        q_nope = torch.matmul(q_nope, q_absorb)

        # ----------FA3-------------
        q_nope = self.fa_quantizer.quant(q_nope, qkv="q")
        compressed_kv = self.fa_quantizer.quant(compressed_kv.unsqueeze(1), qkv="k").squeeze(1)
        _ = self.fa_quantizer.quant(compressed_kv.unsqueeze(1), qkv="v").squeeze(1)
        # ----------FA3-------------

        attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT))
        attn_weights = attn_weights * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is None:
            raise ValueError("Attention mask cannot be None")
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_pe.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
        attn_output = torch.matmul(attn_output, out_absorb.mT)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return new_forward


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

    for name, module in model.named_modules():
        if any(x in module.__class__.__name__ for x in ["Attention", "MMSingleStreamBlock", "MMDoubleStreamBlock"]):
            # 检查是否需要跳过该层
            if any(skip_name in name for skip_name in skip_layers):
                logger.info(f"Skipping FAQuantizer installation for module {name}")
                continue

            # 检查模块是否已经安装了FAQuantizer
            if hasattr(module, "fa_quantizer"):
                logger.warning(f"Module {name} already has FAQuantizer installed.")
                continue

            # 安装FAQuantizer
            module.fa_quantizer = FAQuantizer(config, logger)

            default_attn_type = {
                "deepseekv2": AttentionType.MLA,
                "deepseek_v2": AttentionType.MLA,
                "deepseekv3": AttentionType.MLA,
                "deepseek_v3": AttentionType.MLA,
            }

            # 获取注意力类型：优先使用强制指定的类型
            attn_type = default_attn_type.get(config.model_type, AttentionType.MHA).value

            # 获取并应用forward适配器
            try:
                forward_adapter = ForwardFactory.get_forward_adapter(config.model_type, attn_type)
                module.forward = forward_adapter(module.forward).__get__(module, module.__class__)
                logger.info(f"Successfully installed FAQuantizer for module {name} with attention type {attn_type}")
            except ValueError as e:
                logger.error(f"Failed to install FAQuantizer for module {name}: {str(e)}")
                raise
