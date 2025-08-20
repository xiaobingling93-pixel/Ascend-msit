#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from typing import List, Optional, Tuple
from enum import Enum

import torch
from pydantic import BaseModel
from torch import distributed as dist
from torch.ao.quantization import HistogramObserver as TorchHistogramObserver

from msmodelslim.core import calculate_qparam, fake_quantize
from msmodelslim.core.QAL import QDType, QScope, QStorage
from msmodelslim.utils.exception import (
    SpecError,
    UnexpectedError
)
from msmodelslim.utils.logging import get_logger

# this code is modified from torch.ao.quantization.observer.HistogramObserver._upscale_histogram
def _upscale_histogram(
        histogram: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
        bins: int = 2048,
        upsample_rate: int = 16,
):
    # this turns the histogram into a more fine-coarsed histogram to reduce
    # bin quantization errors
    upsample_rate = max(upsample_rate, 1)
    histogram = histogram.repeat_interleave(upsample_rate) / upsample_rate
    bin_size = (orig_max - orig_min) / (bins * upsample_rate)
    mid_points_histogram = (
            torch.linspace(
                orig_min,
                orig_max,
                bins * upsample_rate + 1,
                device=orig_min.device,
            )[:-1].to(histogram.device)
            + 0.5 * bin_size
    )
    boundaries_new_histogram = torch.linspace(
        update_min, update_max, bins + 1, device=update_min.device
    ).to(histogram.device)
    # this maps the mid-points of the histogram to the new histogram's space
    bucket_assignments = (
            torch.bucketize(mid_points_histogram, boundaries_new_histogram, right=True)
            - 1
    )
    # this then maps the histogram mid-points in the new space, weighted by the original histogram's values
    # this is just the old histogram in the new histogram's space

    # In case due to numerical issues the values land higher/lower than the maximum/minimum
    bucket_assignments[bucket_assignments >= bins] = bins - 1
    bucket_assignments[bucket_assignments < 0] = 0

    update_histogram = torch.bincount(
        bucket_assignments, weights=histogram, minlength=bins
    )
    return update_histogram


def _merge_histogram(
        histogram_list: List[torch.Tensor],
        min_val_list: List[torch.Tensor],
        max_val_list: List[torch.Tensor]
):
    new_min_val = torch.min(torch.stack(min_val_list))
    new_max_val = torch.max(torch.stack(max_val_list))
    histogram_list = [_upscale_histogram(histogram, min_val, max_val, new_min_val, new_max_val) for
                      histogram, min_val, max_val in zip(histogram_list, min_val_list, max_val_list)]
    new_histogram = torch.zeros_like(histogram_list[0])
    for histogram in histogram_list:
        new_histogram += histogram
    return new_histogram, new_min_val, new_max_val


class SearchMethod(str, Enum):
    """搜索方法枚举类"""
    L2_NORM = "l2_norm"           # L2范数搜索
    KL_DIVERGENCE = "kl_divergence"  # KL散度搜索

class HistogramObserverConfig(BaseModel):
    """
    直方图观察器配置类
    
    目前支持两种搜索方法来优化量化参数，包括L2范数、KL散度
    """
    symmetric: bool = False
    search_method: SearchMethod = SearchMethod.L2_NORM
    dtype: QDType = QDType.INT8
    scope: QScope = QScope.PER_TENSOR

    class Config:
        use_enum_values = True


class HistogramObserver(TorchHistogramObserver):
    """
    直方图观察器（HistogramObserver）

    本类用于记录输入张量的直方图分布及其最小/最大值，并据此自动搜索最优的量化截断区间（clip_min/clip_max），
    以便后续计算量化的 scale 和 zero_point。

    主要功能说明：
    基于直方图分布，自动搜索最优的截断区间（clip_min, clip_max），以最小化量化误差    3. 计算量化参数（scale, zero_point），与 MinMaxObserver 类似。

    主要成员变量：
    - config: 直方图观察器配置对象
    - clip_min: 量化截断的最小值，初始化为负无穷
    - clip_max: 量化截断的最大值，初始化为正无穷
    - upsample_rate: 上采样率，用于减少量化误差
    """
    
    def __init__(self, config: HistogramObserverConfig):
        super().__init__(qscheme=torch.per_tensor_affine)
        self.config = config
        self.clip_min = None
        self.clip_max = None   
        self.upsample_rate = 16  

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
        """
        更新直方图，并进行截断值搜索，并保存最佳的量化截断值

        Args:
            x: 输入张量
            sync: 是否同步
            group: 进程组   

        Returns:
            None
            
        Raises:
            InvalidModelError: 当输入张量无效时抛出
        """

        # 输入检测
        if (x is None) or (not isinstance(x, torch.Tensor)):
            raise SpecError(
                "Input must be a valid torch.Tensor",
                action="Please ensure the input is a valid PyTorch tensor"
            )
        
        if x.numel() == 0:
            raise SpecError(
                "Input tensor is empty",
                action="Please check if the input tensor is empty"
            )
        
        if x.isnan().any():
            raise SpecError(
                "Input tensor contains NaN values",
                action="Please check if the input data contains NaN values"
            )
        
        if x.isinf().any():
            raise SpecError(
                "Input tensor contains infinite values",
                action="Please check if the input data contains infinite values"
            )

        # 更新直方图
        x_min, x_max = torch.aminmax(x)
        if x_min == x_max: #torch.histc 不支持min = max的情况，且此时所有值都相同，不需要搜索参数
            get_logger().warning(f"[HistogramObserver] Input tensor is all the same value: {x_min}, skip search.")
            self.clip_min, self.clip_max = x_min, x_max
            return 
        else:
            # torch_npu.histc 不支持bfloat16,转移到cpu
            dtype_support_list = [torch.float, torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8,]
            device = x.device
            if x.dtype  in dtype_support_list:
                self.histogram = self.histogram.to(device=device,dtype=x.dtype)
                self.min_val = self.min_val.to(device=device)
                self.max_val = self.max_val.to(device=device)
                self.forward(x)
            else:
                x = x.to(device='cpu')
                self.histogram = self.histogram.to(device='cpu',dtype=x.dtype)
                self.min_val = self.min_val.to(device='cpu')
                self.max_val = self.max_val.to(device='cpu')
                self.forward(x)
                # 将数据从cpu转移回device
                x = x.to(device=device)
                self.histogram = self.histogram.to(device=device)
                self.min_val = self.min_val.to(device=device)
                self.max_val = self.max_val.to(device=device)
            # 非线性参数搜索
            self.clip_min, self.clip_max = self._non_linear_param_search()
            # 在clip范围内，选择最佳的clip_min,clip_max。在bin_width较大时比较有用。
            mask = (x >= self.clip_min) & (x <= self.clip_max)
            x_clip = x[mask]
            if x_clip.numel() > 0:
                self.clip_min, self.clip_max = torch.aminmax(x_clip)

        #多卡量化 目前尚没有入口
        if sync and group:
            self.forward(x)
            histogram_list = [torch.zeros(self.histogram.shape) for _ in
                              range(dist.get_world_size(group))]
            min_val_list = [torch.zeros(self.min_val.shape) for _ in range(dist.get_world_size(group))]
            max_val_list = [torch.zeros(self.max_val.shape) for _ in range(dist.get_world_size(group))]
            dist.all_gather(histogram_list, self.histogram, group=group)
            dist.all_gather(min_val_list, self.min_val, group=group)
            dist.all_gather(max_val_list, self.max_val, group=group)
            new_histogram, new_min_val, new_max_val = _merge_histogram(histogram_list, min_val_list, max_val_list)
            self.histogram = new_histogram
            self.min_val = new_min_val
            self.max_val = new_max_val
            self.clip_min, self.clip_max = self._non_linear_param_search()

    def reset(self):
        """
        重置直方图观察器
        """
        self.clip_min = None
        self.clip_max = None
        self.min_val = torch.tensor(float("inf"), device=self.histogram.device) 
        self.max_val = torch.tensor(float("-inf"), device=self.histogram.device)
        self.histogram = torch.zeros(self.bins, device=self.histogram.device)

    def get_clip_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回用于量化截断的上下界（clip bounds），并非真实的最小最大值，而是通过直方图搜索得到、能减小量化误差的截断范围。
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (clip_min, clip_max) 截断的上下界
            
        Raises:
            SpecError: 当clip_min或clip_max未设置时抛出
        """
        if self.clip_min is None or self.clip_max is None:
            raise SpecError(
                "Clip min or clip max is not set.",
                action=" Please call update first."
            )
        
        # 处理无穷值情况
        finfo_dtype = torch.finfo(self.clip_min.dtype)
        if torch.isinf(self.clip_min) :
            self.clip_min = torch.tensor(finfo_dtype.min)
        if torch.isinf(self.clip_max):
            self.clip_max = torch.tensor(finfo_dtype.max)
        
        return self.clip_min, self.clip_max
    
    def _compute_quantization_error(self, start_bin: int, end_bin: int) -> float:
        """
        选择量化方法，计算量化误差，目前支持L2范数，KL散度
        
        Args:
            start_bin: 起始bin索引
            end_bin: 结束bin索引
            
        Returns:
            float: 量化误差值
            
        Raises:
            UnsupportedError: 当使用KL散度但未实现时抛出
        """
        method = self.config.search_method
        
        if method == SearchMethod.L2_NORM:
            return self._compute_l2_error(start_bin, end_bin)
        elif method == SearchMethod.KL_DIVERGENCE:
            return self._compute_kl_error(start_bin, end_bin)



    def _compute_kl_error(self, next_start_bin: int, next_end_bin: int) -> float:
        """
        计算KL散度误差
        
        Args:
            next_start_bin: 起始bin索引
            next_end_bin: 结束bin索引
            
        Returns:    
            float: KL散度误差

        算法原理：
        使用KL散度作为量化误差，计算候选分布与真实分布的差异。
        1.计算量化后的分布。
        在得到原始分布的直方图后，以每个bin的中间点作为量化点，进行伪量化，计算量化后的分布。
        2.计算候选分布。
        由于伪量化后，直方图变得稀疏，出现大量为0的bin_fakequant，直接计算KL散度会导致log0问题。
        如果真实分布中，bin_true就是0.此时让量化分布加上一个极小值，避免log0问题即可。此时KL散度为0，符合预期。
        更多情况下，bin_fakequant是因为量化导致丢失了分布信息。此时，选用均匀分布作为"量化到同一个bin_quant的bin_true"的分布，
        将bin_fakequant的概率设为bin_quant的概率除以对应bin_true的数量。
        3.计算KL散度。
        KL = sum(p_i * log(p_i / q_i)), p为真实分布，q为量化分布。
        """
        eps = self.eps.to(self.histogram.device)
        bin_width = (self.max_val.item()/ self.bins - self.min_val.item()/ self.bins) 

        # 计算真实分布
        true_dist = self.histogram / self.histogram.sum()

        # 计算直方图的中间点
        quant_mid_bin = torch.arange(0.5, self.bins + 0.5, device=self.histogram.device)
        quant_mid_points = quant_mid_bin * bin_width + self.min_val

        # 模拟量化过程
        # 计算scale和zero_point
        quant_min_val = next_start_bin * bin_width + self.min_val
        quant_max_val = (next_end_bin + 1) * bin_width + self.min_val
        q_param = calculate_qparam(
            min_val=quant_min_val,      # 最小截断值
            max_val=quant_max_val,      # 最大截断值
            q_dtype=QDType(self.config.dtype),    
            q_scope=QScope(self.config.scope),      
            symmetric=self.config.symmetric,       
        )
        # 计算伪量化后,原直方图每个bin_true对应的新bin_fakequant
        fake_quantized_dist = fake_quantize(QStorage(dtype=QDType.FLOAT, value=quant_mid_points), q_param).value
        fake_quantized_dist[:next_start_bin] = quant_min_val
        fake_quantized_dist[next_end_bin:] = quant_max_val
        fake_quantized_dist = ((fake_quantized_dist-quant_min_val)//bin_width + next_start_bin).clamp(next_start_bin,next_end_bin).int()

        # 计算候选分布
        candidate_dist = torch.zeros_like(self.histogram).float()
        # 将每个quant_dist[i]均匀分配到fake_quantized_dist[i]和fake_quantized_dist[i+1]之间的所有bin
        # 计算在平均之前，bin_fakequant的分布
        fake_quant_dict = torch.bincount(fake_quantized_dist, weights=true_dist.to(torch.float16), minlength=self.bins)
        # 计算每个bin_fakequant对应的bin_true数量
        fake_quant_dist = torch.bincount(fake_quantized_dist, minlength=self.bins)
        # 进行平均
        fake_quant_dict = fake_quant_dict / (fake_quant_dist + eps)
        bin_indices = fake_quantized_dist.long()  # 确保索引为整数类型
        candidate_dist.add_(fake_quant_dict[bin_indices])
        
        # 计算KL散度
        kl_div = torch.sum(true_dist * torch.log((true_dist+eps) / (candidate_dist+eps)))
        return kl_div.item()

    def _compute_l2_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        这一段代码来自torch.ao.quantization.observer.HistogramObserver._compute_quantization_error
        
        计算L2范数误差。
        """
        bin_width = (self.max_val.item()/ self.bins - self.min_val.item()/ self.bins) 

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(
            delta_begin,
            torch.ones(self.bins, device=self.histogram.device) * delta_end,
            density,
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        这一段代码来自torch.ao.quantization.observer.HistogramObserver._non_linear_param_search
        非线性参数搜索。

        该方法用于通过最小化量化误差来选择min/max截断值。
        通过选择新的min/max，可以过滤输入分布中的离群值（outlier）。
        搜索方法为：
        每次将下界和上界移动固定的百分位数，计算量化误差，直到量化误差不再变小，或者下界超过了上界。
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (new_min, new_max) 搜索得到的最优截断值
            
        Raises:
            UnexpectedError: 当搜索过程中发生意外错误时抛出
        """
        try:
            if self.histogram.size()[0] != self.bins:
                raise UnexpectedError(
                    f"Histogram bins mismatch: expected {self.bins}, got {self.histogram.size()[0]}",
                    action="Please check if the histogram configuration is correct"
                )
            bin_width = (self.max_val / self.bins - self.min_val / self.bins)  # 避免溢出

            # cumulative sum
            total = torch.sum(self.histogram).item()

            # 直方图为空，不进行搜索
            if total == 0:
                get_logger().warning("Histogram is empty, skipping search. This may be caused by an empty input tensor or the input tensor's range being too large.")
                return self.min_val, self.max_val
            cSum = torch.cumsum(self.histogram, dim=0)

            stepsize = 1e-5  # granularity
            alpha = 0.0  # lower bound
            beta = 1.0  # upper bound
            start_bin = 0
            end_bin = self.bins - 1
            norm_min = self._compute_quantization_error(start_bin, end_bin)

            while alpha < beta:
                # Find the next step
                next_alpha = alpha + stepsize
                next_beta = beta - stepsize

                # find the left and right bins between the quantile bounds
                l = start_bin
                r = end_bin
                while l < end_bin and cSum[l] < next_alpha * total:
                    l = l + 1
                while r > start_bin and cSum[r] > next_beta * total:
                    r = r - 1

                # decide the next move
                next_start_bin = start_bin
                next_end_bin = end_bin
                if (l - start_bin) > (end_bin - r):
                    # move the start bin
                    next_start_bin = l
                    alpha = next_alpha
                else:
                    # move the end bin
                    next_end_bin = r
                    beta = next_beta

                if next_start_bin == start_bin and next_end_bin == end_bin:
                    continue

                # calculate the quantization error using next_start_bin and next_end_bin
                norm = self._compute_quantization_error(next_start_bin, next_end_bin)

                if norm > norm_min:
                    break
                norm_min = norm
                start_bin = next_start_bin
                end_bin = next_end_bin

            new_min = self.min_val + bin_width * start_bin
            new_max = (end_bin + 1) * (self.min_val / (end_bin + 1) + bin_width)  # 防溢出
            return new_min, new_max
            
        except Exception as e:
            raise UnexpectedError(
                f"Unexpected error during non-linear parameter search: {e}",
                action="Please check if the histogram data is valid"
            )
