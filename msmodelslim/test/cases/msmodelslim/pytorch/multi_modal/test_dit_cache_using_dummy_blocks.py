import logging
import unittest

import torch
import torch.nn as nn

from msmodelslim.pytorch.multi_modal.dit_cache.adaptor import DitCacheSearchConfig, DitCacheAdaptor, DitCacheDummy, \
    DitCacheConfig


# ----------------- 构造简易的 pipeline 与 transformer 模块 -----------------
class DummyBlock(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.t_idx = -1

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask,
                timestep, cross_attention_kwargs, class_labels, frame, height, width):
        """
        为便于观察，这里仅返回一个计算值：
        hidden_states + self.idx + timestep
        其他参数仅作为占位存在
        """
        return hidden_states + torch.tensor([self.t_idx, self.idx])


class DummyTransformer(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        # 构造 num_blocks 个 DummyBlock
        self.transformer_blocks = nn.ModuleList([DummyBlock(i) for i in range(num_blocks)])


class DummyPipeline:
    """
    模拟一个简单的 pipeline，包含 transformer 模块
    """

    def __init__(self, num_blocks):
        self.name = "DummyPipeline"
        self.transformer = DummyTransformer(num_blocks)

    def __call__(self, *args, **kwargs):
        # 定义各个参数的 dummy 值
        attention_mask = torch.tensor(1.0)
        encoder_hidden_states = torch.tensor(1.0)
        encoder_attention_mask = torch.tensor(1.0)
        cross_attention_kwargs = {}
        class_labels = 0
        frame = 0
        height = 1
        width = 1

        output = None
        # 模拟 5 个时间步的前向传递
        for t in range(5):
            print(f"Time step {t}:")
            hidden_states = torch.tensor([0.0, 0.0])
            input_val = hidden_states
            for i, block in enumerate(self.transformer.transformer_blocks):
                block.t_idx = t
                DitCacheAdaptor.set_timestep_idx(t)

                output = block.forward(
                    input_val,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=t * 0.1,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame,
                    height=height,
                    width=width,
                )
                print(f"  Block {i} output: {output}")
                input_val = output  # 将当前 block 输出作为下一个 block 的输入
            print("------")
        print("----- End of System Test -----\n")

        if output is None:
            raise ValueError("No output found")
        return output


# 为了让类型提示通过，这里将 OpenSoraPipelineV1_2 定义为 DummyPipeline 的别名
OpenSoraPipelineV1_2 = DummyPipeline


# ----------------- 系统测试 (ST) -----------------
def run_forward_test():
    print("----- Running System Test -----")
    num_blocks = 5  # 假设 transformer 有 4 个 block
    pipeline = DummyPipeline(num_blocks)
    # 配置 DitCache：指定总 block 数和采样步数

    config = DitCacheSearchConfig(dit_block_num=num_blocks, num_sampling_steps=5)

    dit_cache_config = DitCacheConfig(
        # Timestep start：开始缓存的时间步
        cache_step_start=1,
        # Timestep interval：每隔 n 个时间步计算一次，其他复用 cache
        cache_step_interval=3,
        # Block 缓存起始索引 idx， 若值为 0 表示第一个 block 开始
        cache_block_start=1,
        # Block 设定缓存区域 block 数量：
        cache_num_blocks=3
    )

    cache_adaptor = DitCacheAdaptor(pipeline, config)
    cache_adaptor.dit_cache_config = dit_cache_config

    # 定义各个参数的 dummy 值
    attention_mask = torch.tensor(1.0)
    encoder_hidden_states = torch.tensor(1.0)
    encoder_attention_mask = torch.tensor(1.0)
    cross_attention_kwargs = {}
    class_labels = 0
    frame = 0
    height = 1
    width = 1

    # 模拟 5 个时间步的前向传递
    for t in range(5):
        print(f"Time step {t}:")
        hidden_states = torch.tensor([0.0, 0.0])
        input_val = hidden_states
        for i, block in enumerate(pipeline.transformer.transformer_blocks):
            block.t_idx = t
            DitCacheAdaptor.set_timestep_idx(t)

            output = block.forward(
                input_val,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t * 0.1,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                frame=frame,
                height=height,
                width=width,
            )
            print(f"  Block {i} output: {output}")
            input_val = output  # 将当前 block 输出作为下一个 block 的输入
        print("------")
    print("----- End of System Test -----\n")


def run_search_test():
    print("----- Running System Test -----")
    num_blocks = 5  # 假设 transformer 有 4 个 block
    pipeline = DummyPipeline(num_blocks)
    # 配置 DitCache：指定总 block 数和采样步数

    config = DitCacheSearchConfig(dit_block_num=num_blocks, num_sampling_steps=5)
    cache_adaptor = DitCacheAdaptor(pipeline, config)

    def run_pipeline_and_save_videos(pipeline: DummyPipeline):
        tensor_ouput = pipeline('default input')
        vid = torch.zeros((30, 64, 64, 3)) + tensor_ouput[0]
        return [vid.to(torch.uint8).cpu().numpy()]

    cache_adaptor.search(run_pipeline_and_save_videos, prompts_num=1)


import os
import torch.distributed as dist


class TestDitCacheAdaptor(unittest.TestCase):
    def setUp(self):
        # 设置分布式环境变量
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

        # 初始化分布式进程组
        dist.init_process_group(
            backend='hccl',  # 单机建议使用gloo，GPU用nccl
            init_method='tcp://localhost:29599',
            rank=0,
            world_size=1
        )

    def tearDown(self):
        # 释放分布式资源
        dist.destroy_process_group()

    def test_search_with_dit_cache(self):
        run_search_test()

    def test_forward_with_dit_cache(self):
        run_forward_test()


# ----------------- 主入口 -----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # 先运行系统测试，观察缓存适配器在多步前向传递中的效果
    # run_system_test()
    # 再运行单元测试
    unittest.main(verbosity=2)
