#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import partial
from typing import Dict, Callable, List, Any, Literal
from dataclasses import field

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict

from msmodelslim.core.api import iter_smooth
from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.QAL.qtypes import (
    LinearLinearSubgraph,
    NormLinearSubgraph,
    RMSNormBias,
    IterSmoothConfig,
    UpDownSubgraph,
    OVSubgraph
)
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.anti_outlier.base import StatKey
from msmodelslim.quant.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.dist import DistHelper
from msmodelslim.quant.processor.anti_outlier.base import BaseSmoothProcessor, VirtualVModule
from msmodelslim.model.adapter_types import SubgraphInfo
from msmodelslim.quant.processor.anti_outlier.base import GraphOpt
from msmodelslim.quant.processor.anti_outlier.base import SubgraphProcessor
from msmodelslim.utils.logging import get_logger


class M4ProcessorConfig(AutoProcessorConfig):
    type: Literal["iter_smooth"] = "iter_smooth"
    alpha: float = 0.9
    scale_min: float = 1e-5
    symmetric: bool = False
    enable_subgraph_type: List[str] = field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"]
    )
    include: List[str] = []
    exclude: List[str] = []
    # 子图处理优先级配置（数字越小优先级越高）
    # 使用Field的frozen=True确保此字段不可被外部更改
    subgraph_priority: Dict[str, int] = Field(
        default_factory=lambda: {
            "up-down": 1,        # 最高优先级：MLP门控机制
            "ov": 2,            # 中等优先级：注意力机制
            "norm-linear": 3,   # 较低优先级：归一化层
            "linear-linear": 4  # 最低优先级：线性层
        },
        frozen=True  # 单独为subgraph_priority字段启用frozen
    )


@QABCRegistry.register(dispatch_key=M4ProcessorConfig, abc_class=AutoSessionProcessor)
class M4Processor(BaseSmoothProcessor):
    def __init__(self, model: nn.Module, config: M4ProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        self.config = config
        self.validate_parameters()
        self.act_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.dist_helper = DistHelper(self.model) if dist.is_initialized() else None
        
        # 设置anti_method属性
        setattr(self.model, 'anti_method', 'm4')
        
        # 存储hook句柄，用于后续删除
        self.hook_handles = {}
    
    def __del__(self):
        """
        析构方法，确保在对象销毁时清理所有hook
        """
        try:
            self._remove_all_hooks()
        except Exception:
            pass  # 析构时忽略异常

    def validate_parameters(self):
        """
        验证所有参数的合法性
        """
        # 验证 alpha 参数
        if not isinstance(self.config.alpha, float) or self.config.alpha <= 0:
            raise ValueError(f"alpha 必须是大于0的数值，当前值: {self.config.alpha} (类型: {type(self.config.alpha)})")

        # 验证 scale_min 参数
        if not isinstance(self.config.scale_min, float) or self.config.scale_min <= 0:
            raise ValueError(f"scale_min 必须是大于0的数值，当前值: {self.config.scale_min} (类型: {type(self.config.scale_min)})")

        # 验证 symmetric 参数
        if not isinstance(self.config.symmetric, bool):
            raise ValueError(f"symmetric 必须是布尔类型，当前值: {self.config.symmetric} (类型: {type(self.config.symmetric)})")

        # 验证 enable_subgraph_type 参数
        if not isinstance(self.config.enable_subgraph_type, list):
            raise ValueError(
                f"enable_subgraph_type 必须是列表类型，当前值: {self.config.enable_subgraph_type} (类型: {type(self.config.enable_subgraph_type)})")

        # 验证 enable_subgraph_type 中的元素
        valid_subgraph_types = ["norm-linear", "linear-linear", "ov", "up-down"]
        for subgraph_type in self.config.enable_subgraph_type:
            if not isinstance(subgraph_type, str):
                raise ValueError(
                    f"enable_subgraph_type 中的元素必须是字符串类型，当前元素: {subgraph_type} (类型: {type(subgraph_type)})")
            if subgraph_type not in valid_subgraph_types:
                raise ValueError(
                    f"enable_subgraph_type 中的元素必须在 {valid_subgraph_types} 中，当前元素: {subgraph_type}")

        # 验证 include 参数
        if not isinstance(self.config.include, list):
            raise ValueError(f"include 必须是列表类型，当前值: {self.config.include} (类型: {type(self.config.include)})")

        # 验证 include 中的元素
        for item in self.config.include:
            if not isinstance(item, str):
                raise ValueError(f"include 中的元素必须是字符串类型，当前元素: {item} (类型: {type(item)})")

        # 验证 exclude 参数
        if not isinstance(self.config.exclude, list):
            raise ValueError(f"exclude 必须是列表类型，当前值: {self.config.exclude} (类型: {type(self.config.exclude)})")

        # 验证 exclude 中的元素
        for item in self.config.exclude:
            if not isinstance(item, str):
                raise ValueError(f"exclude 中的元素必须是字符串类型，当前元素: {item} (类型: {type(item)})")

    def support_distributed(self) -> bool:
        return True

    def preprocess(self, request: BatchProcessRequest) -> None:
        adapter_config = self.adapter.get_adapter_config_for_subgraph()
        smooth_processor = SubgraphProcessor(model=self.model, adapter_config=adapter_config, m4_config=self.config)
        # 获取全量子图
        self.subgraph_info = smooth_processor.get_global_subgraph_info()
        # 根据配置过滤子图
        self.subgraph_info = smooth_processor.find_subgraphs_by_config(
            self.subgraph_info, 
            self.config, 
            request.name
        )
        
        # 遍历 subgraph_info，将 norm_module 替换为 RMSNormBias
        for subgraph in self.subgraph_info:
            if subgraph.subgraph_type == "norm-linear" and subgraph.metadata:
                # 获取 norm_module 信息
                norm_name = subgraph.metadata.get('source_name')
                norm_module = subgraph.metadata.get('source_module')
                
                if norm_name and norm_module is not None:
                    try:
                        # 检查 norm_module 是否有 weight 属性
                        if hasattr(norm_module, 'weight'):
                            # 创建 RMSNormBias 实例
                            norm_bias = RMSNormBias(norm_module.weight.shape[-1])
                            norm_bias.weight.data.copy_(norm_module.weight.data)
                            norm_bias.weight.data = norm_bias.weight.data.type(norm_module.weight.data.dtype)
                            if hasattr(norm_module, 'bias') and norm_module.bias is not None:
                                norm_bias.bias.data.copy_(norm_module.bias.data)
                                norm_bias.bias.data = norm_bias.bias.data.type(norm_module.weight.data.dtype)
                            norm_bias.to(norm_module.weight.data.device)
                            GraphOpt.set_module(self.model, norm_name, norm_bias)
                            subgraph.metadata['source_module'] = norm_bias
                            get_logger().debug(f"{norm_name}: {type(norm_module)} -> {type(norm_bias)}")
                        else:
                            get_logger().warning(f"Norm module {norm_name} does not have weight attribute")
                    except Exception as e:
                        get_logger().warning(f"Failed to replace norm module {norm_name}: {e}")

        get_logger().info(f"[Smooth] Processed {len(self.subgraph_info)} subgraphs for submodule {request.name}")
        return super().preprocess(request)

    def _get_stats_hook(self, name: str) -> Callable:
        def stats_hook(name: str, module: nn.Linear, args: tuple, kwargs: dict) -> None:
            tensor = args[0]

            if name not in self.act_stats:
                self.act_stats[name] = {}
                # 存储收集的tensor到CPU，避免OOM
                self.act_stats[name][StatKey.TENSOR] = tensor.cpu()

            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()  # [N,C]
            
            if self.dist_helper is not None and self.dist_helper.is_shared(name):
                tensor = torch.cat(self.dist_helper.gather_variable_shapes(tensor), dim=0)
            coming_max = torch.max(tensor, dim=0)[0]  # [C]
            coming_min = torch.min(tensor, dim=0)[0]  # [C]

            statis_dict = self.act_stats[name]

            # collect the min-max value
            if StatKey.STAT_KEY_MAX in statis_dict:
                statis_dict[StatKey.STAT_KEY_MAX] = torch.max(statis_dict[StatKey.STAT_KEY_MAX], coming_max)  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_MAX] = coming_max

            if StatKey.STAT_KEY_MIN in statis_dict:
                statis_dict[StatKey.STAT_KEY_MIN] = torch.min(statis_dict[StatKey.STAT_KEY_MIN], coming_min)  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_MIN] = coming_min

            # channel shifting
            if StatKey.STAT_KEY_SHIFT in statis_dict:
                statis_dict[StatKey.STAT_KEY_SHIFT] = (statis_dict[StatKey.STAT_KEY_MAX] + statis_dict[
                    StatKey.STAT_KEY_MIN]) / 2  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_SHIFT] = (coming_max + coming_min) / 2

            channel_max = torch.max(tensor.abs().detach(), dim=0)[0]

            if StatKey.STAT_KEY_SMOOTH_SCALE in statis_dict:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = torch.max(statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE],
                                                                       channel_max)
            else:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return partial(stats_hook, name)

    def _install_statis_hook(self, name: str, module: nn.Module) -> None:
        """
        为所有子图中的linear模块安装统计钩子
        
        Args:
            name: 模块名称
            module: 目标模块
        """
        for subgraph in self.subgraph_info:
            if not subgraph.metadata:
                continue
                
            # 根据子图类型获取需要安装钩子的模块名称
            target_names = self._get_target_names_for_hook(subgraph)
            
            # 为每个目标模块安装钩子
            for target_name in target_names:
                if target_name:
                    self._install_hook_for_module(target_name)
    
    def _get_target_names_for_hook(self, subgraph: SubgraphInfo) -> List[str]:
        """
        根据子图类型获取需要安装钩子的模块名称列表
        
        Args:
            subgraph: 子图信息
            
        Returns:
            List[str]: 目标模块名称列表
        """
        subgraph_type = subgraph.subgraph_type
        metadata = subgraph.metadata
        
        if subgraph_type == "norm-linear" or subgraph_type == "up-down":
            # Norm-Linear: 为所有target_names安装钩子
            return metadata.get('target_names', [])
            
        if subgraph_type == "linear-linear" or subgraph_type == "ov":
            # Linear-Linear: 为linear2模块安装钩子
            target_name = metadata.get('target_names', [''])[0]
            return [target_name] if target_name else []
        
        # 默认情况：返回空列表
        return []
    
    def _install_hook_for_module(self, module_name: str) -> None:
        """
        为指定模块安装统计钩子
        
        Args:
            module_name: 模块名称
        """
        try:
            module = self.model.get_submodule(module_name)
            if isinstance(module, nn.Linear):
                # 保存hook句柄，用于后续删除
                hook_handle = module.register_forward_hook(self._get_stats_hook(module_name))
                self.hook_handles[module_name] = hook_handle
                get_logger().debug(f"成功为模块 {module_name} 安装统计钩子")
            else:
                get_logger().warning(f"模块 {module_name} 不是Linear类型，跳过钩子安装")
        except Exception as e:
            get_logger().warning(f"为模块 {module_name} 安装统计钩子失败: {e}")

    def _apply_smooth(self, name: str, module: nn.Module) -> None:
        """
        应用平滑处理到所有子图（按优先级顺序）
        
        Args:
            name: 模块名称
            module: 目标模块
        """
        get_logger().info(f"开始应用M4平滑处理到模块: {name}")
        
        # 按优先级顺序处理子图：up-down -> ov -> norm-linear -> linear-linear
        self._process_subgraphs_by_priority()
        
        # 清理统计信息
        self.act_stats.clear()
        
        # 删除所有已安装的hook
        self._remove_all_hooks()
        
        get_logger().info(f"完成M4平滑处理，清理了统计信息和hook")
    
    def _remove_all_hooks(self) -> None:
        """
        删除所有已安装的hook
        """
        for module_name, hook_handle in self.hook_handles.items():
            try:
                hook_handle.remove()
                get_logger().debug(f"成功删除模块 {module_name} 的hook")
            except Exception as e:
                get_logger().warning(f"删除模块 {module_name} 的hook失败: {e}")
        
        # 清空hook句柄字典
        self.hook_handles.clear()
        get_logger().info(f"已删除所有hook")

    def _process_subgraphs_by_priority(self) -> None:
        """
        按优先级顺序处理子图
        
        优先级顺序（可配置）：
        1. up-down (最高优先级，MLP门控机制)
        2. ov (中等优先级，注意力机制)
        3. norm-linear (较低优先级，归一化层)
        4. linear-linear (最低优先级，线性层)
        """
        # 使用配置中的优先级设置
        priority_order = self.config.subgraph_priority
        
        # 按优先级排序子图
        sorted_subgraphs = sorted(
            self.subgraph_info,
            key=lambda x: priority_order.get(x.subgraph_type, 999)  # 未知类型优先级最低
        )
        
        get_logger().info(f"按优先级排序后的子图处理顺序:")
        for i, subgraph in enumerate(sorted_subgraphs):
            priority = priority_order.get(subgraph.subgraph_type, 999)
            get_logger().info(f"  {i+1}. {subgraph.subgraph_type} (优先级: {priority}) - {subgraph.name}")
        
        # 按优先级顺序处理子图
        for subgraph in sorted_subgraphs:
            try:
                priority = priority_order.get(subgraph.subgraph_type, 999)
                get_logger().debug(
                    f"处理子图: {subgraph.subgraph_type} - {subgraph.name} "
                    f"(优先级: {priority})"
                )
                self._process_single_subgraph(subgraph)
            except Exception as e:
                get_logger().error(f"处理子图 {subgraph.name} 时发生错误: {e}")
                continue
    
    def _process_single_subgraph(self, subgraph: SubgraphInfo) -> None:
        """
        处理单个子图
        
        Args:
            subgraph: 子图信息
        """
        subgraph_type = subgraph.subgraph_type
        metadata = subgraph.metadata or {}
        
        get_logger().debug(
            f"处理子图类型: {subgraph_type}, 名称: {subgraph.name}"
        )
        common_metadata = self._extract_common_metadata(metadata)
        # 根据子图类型调用相应的处理方法
        if subgraph_type == "norm-linear":
            self._apply_norm_linear_smooth(common_metadata)
        elif subgraph_type == "linear-linear":
            self._apply_linear_linear_smooth(common_metadata)
        elif subgraph_type == "ov":
            self._apply_ov_smooth(common_metadata)
        elif subgraph_type == "up-down":
            self._apply_up_down_smooth(common_metadata)
        else:
            get_logger().warning(f"不支持的子图类型: {subgraph_type}")
    
    def _extract_common_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取公共的metadata信息
        
        Args:
            metadata: 原始元数据
            
        Returns:
            Dict[str, Any]: 提取的公共信息
        """
        return {
            'source_name': metadata.get('source_name', ''),
            'source_module': metadata.get('source_module'),
            'target_modules': metadata.get('target_modules', []),
            'target_names': metadata.get('target_names', []),
            'fusion_flag': metadata.get('fusion_flag', False),
            'num_attention_heads': metadata.get('num_attention_heads'),
            'num_key_value_heads': metadata.get('num_key_value_heads')
        }

    def _apply_norm_linear_smooth(self, common_metadata: Dict[str, Any]) -> None:
        """应用Norm-Linear平滑"""
        # 验证模块类型
        if not isinstance(common_metadata['source_module'], RMSNormBias):
            get_logger().warning(f"Norm模块 {common_metadata['source_name']} 不是RMSNormBias类型，跳过")
            return
        
        # 应用平滑
        self._apply_smooth_to_subgraph(
            NormLinearSubgraph(common_metadata['source_module'], common_metadata['target_modules']),
            common_metadata['target_modules']
        )
    
    def _apply_linear_linear_smooth(self, common_metadata: Dict[str, Any]) -> None:
        """应用Linear-Linear平滑"""        # 获取第一个目标模块（Linear-Linear通常只有一个目标）
        target_module = common_metadata['target_modules'][0] if common_metadata['target_modules'] else None
        
        if not target_module:
            get_logger().warning("Linear-Linear子图缺少目标模块")
            return
        
        # 应用平滑
        self._apply_smooth_to_subgraph(
            LinearLinearSubgraph(common_metadata['source_module'], target_module),
            [target_module]
        )
    
    def _apply_ov_smooth(self, common_metadata: Dict[str, Any]) -> None:
        """应用OV平滑（输出-值投影）"""
        # 获取OV特定的信息
        o_name = common_metadata['target_names'][0] if common_metadata['target_names'] else ''
        v_name = common_metadata['source_name']
        o_module = common_metadata['target_modules'][0] if common_metadata['target_modules'] else None
        v_module = common_metadata['source_module']
        fusion_flag = common_metadata['fusion_flag']
        
        # 获取注意力头数量
        num_attention_heads = common_metadata['num_attention_heads']
        num_key_value_heads = common_metadata['num_key_value_heads']
        
        if num_attention_heads is None or num_key_value_heads is None:
            num_attention_heads = self.get_num_attention_heads()
            num_key_value_heads = self.get_num_key_value_heads()
        
        if not o_name or not v_name:
            get_logger().warning(f"OV子图缺少必要的名称信息: o_name={o_name}, v_name={v_name}")
            return
        
        try:            
            if not o_module or not v_module:
                get_logger().warning(f"无法获取OV子图的模块: o_module={o_module is not None}, v_module={v_module is not None}")
                return
            
            if not isinstance(o_module, nn.Linear):
                get_logger().warning(f"O模块 {o_name} 不是Linear类型，跳过")
                return
            
            # 根据融合标志选择平滑方法
            if fusion_flag:
                self._apply_qkv_fusion_smooth(v_module, o_module, v_name, o_name, 
                                           num_attention_heads, num_key_value_heads)
            else:
                self._apply_standard_ov_smooth(v_module, o_module, v_name, o_name,
                                              num_attention_heads, num_key_value_heads)
                
        except Exception as e:
            get_logger().error(f"应用OV平滑时发生错误: {e}")
    
    def _apply_qkv_fusion_smooth(self, v_module: nn.Module, o_module: nn.Module, 
                                v_name: str, o_name: str, 
                                num_attention_heads: int, num_key_value_heads: int) -> None:
        """
        应用QKV融合平滑
        
        Args:
            v_module: V投影模块
            o_module: O投影模块
            v_name: V模块名称
            o_name: O模块名称
            num_attention_heads: 注意力头数量
            num_key_value_heads: 键值头数量
        """
        if not isinstance(v_module, nn.Linear):
            get_logger().warning(f"V模块 {v_name} 不是Linear类型，跳过QKV融合")
            return
        
        # 创建虚拟V模块
        virtual_v_module = VirtualVModule(v_module, num_attention_heads, num_key_value_heads)
        
        # 应用平滑
        self._apply_smooth_to_subgraph(
            OVSubgraph(
                o_proj=o_module,
                v_proj=virtual_v_module,
                num_attention_heads=num_attention_heads,
                key_value_heads=num_key_value_heads
            ),
            [o_module]
        )
        
        # 更新原始QKV模块权重
        virtual_v_module.update_qkv_weights()
        
        get_logger().debug(f"成功应用QKV融合平滑: {v_name} -> {o_name}")
    
    def _apply_standard_ov_smooth(self, v_module: nn.Module, o_module: nn.Module,
                                 v_name: str, o_name: str,
                                 num_attention_heads: int, num_key_value_heads: int) -> None:
        """
        应用标准OV平滑
        
        Args:
            v_module: V投影模块
            o_module: O投影模块
            v_name: V模块名称
            o_name: O模块名称
            num_attention_heads: 注意力头数量
            num_key_value_heads: 键值头数量
        """
        if not isinstance(v_module, nn.Linear):
            get_logger().warning(f"V模块 {v_name} 不是Linear类型，跳过标准OV平滑")
            return
        
        # 应用平滑
        self._apply_smooth_to_subgraph(
            OVSubgraph(
                o_proj=o_module,
                v_proj=v_module,
                num_attention_heads=num_attention_heads,
                key_value_heads=num_key_value_heads
            ),
            [o_module]
        )
        
        get_logger().debug(f"成功应用标准OV平滑: {v_name} -> {o_name}")
    
    def _apply_up_down_smooth(self, common_metadata: Dict[str, Any]) -> None:
        """应用Up-Down平滑（MLP门控机制）"""
        # 获取Up-Down特定的模块
        up_module = common_metadata['source_module']
        down_module = common_metadata['target_modules'][0] if len(common_metadata['target_modules']) > 0 else None
        gate_module = common_metadata['target_modules'][1] if len(common_metadata['target_modules']) > 1 else None
        
        if not all([up_module, down_module]):
            get_logger().warning(f"Up-Down子图缺少必要的模块信息: up={up_module is not None}, down={down_module is not None}")
            return
        
        # 应用平滑
        self._apply_smooth_to_subgraph(
            UpDownSubgraph(up_module, down_module, gate_module),
            [down_module]
        )
    
    def _apply_smooth_to_subgraph(self, subgraph_obj: Any, linear_modules: List[nn.Module]) -> None:
        """
        通用的平滑应用方法
        
        Args:
            subgraph_obj: 子图对象
            linear_modules: 线性模块列表
        """
        # 构建SmoothContext
        smooth_context = self._build_smooth_context(linear_modules)
        
        # 创建平滑配置
        smooth_quant_cfg = IterSmoothConfig(
            alpha=self.config.alpha,
            shift=self.config.symmetric,
            scale_min=self.config.scale_min
        )
        
        # 应用平滑
        iter_smooth(subgraph_obj, smooth_quant_cfg, smooth_context)
