# 多模态生成模型量化

## 模型介绍

[SD3](https://stability.ai/news/stable-diffusion-3) Stable Diffusion 3, 由stability.ai发布的强大的文本到图像模型，在多主题提示、图像质量和拼写功能方面的性能得到了大幅提升。

[Open-Sora-Plan v1.2](https://github.com/PKU-YuanGroup/Open-Sora-Plan) 是一个开源的多模态视频生成模型，由北大-兔展AIGC联合实验室共同发起，专注于高效视频生成任务。

[Flux.1](https://blackforestlabs.io/flux-1/)是由 Black Forest Labs 开发的一款开源的 120 亿参数的图像生成模型，它能够根据文本描述生成高质量的图像。

[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) 是腾讯发布的一种新颖的开源视频基础模型，它在视频生成方面的性能可与领先的闭源模型相媲美，甚至优于领先的闭源模型。

## 已验证量化模型
表中模型链接为对应权重地址。
| 模型 | 支持量化 | 权重链接
|-----------|-----------|---------|
| SD3-Medium | W8A8静态量化 | [link](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
| Open-Sora-Plan v1.2 | W8A8静态量化 | [link](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0)
| FLUX.1-dev | W8A8静态量化，W8A8分时间步量化，FA3+W8A8动态量化，异常值抑制+W8A8动态量化 | [link](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)
| HunyuanVideo | W8A8静态量化，W8A8分时间步量化，FA3+W8A8动态量化，异常值抑制+W8A8动态量化 | [link](https://huggingface.co/tencent/HunyuanVideo)

## 环境配置
- 配套CANN版本请选择8.2.RC1及之后的版本
- 具体环境配置请参考[使用说明](../../README.md)
- 当前多模态生成模型统一接口依赖于pydantic库
  - pip install pydantic
- SD3-Medium依赖于diffusers库
  - pip install -U diffusers
- Open-Sora-Plan v1.2相关环境配置参考[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)
  - 参考 [open_sora_planv1_2 reamde](https://modelers.cn/models/MindIE/open_sora_planv1_2) 安装浮点模型的环境依赖，并确保浮点推理能正常运行
  - pip install huggingface_hub==0.25.2
- Flux.1-dev相关环境配置参考[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)
  - 参考 [Flux reamde](https://modelers.cn/models/MindIE/FLUX.1-dev) 安装浮点模型的环境依赖，并确保浮点推理能正常运行
- HunyuanVideo相关环境配置参考[MindIE/hunyuan_video](https://modelers.cn/models/MindIE/hunyuan_video)
  - 参考 [HunyuanVideo reamde](https://modelers.cn/models/MindIE/hunyuan_video) 安装浮点模型的环境依赖，并确保浮点推理能正常运行

## 使用案例
使用量化前，需要加载模型和校准数据，其中加载模型依赖于diffusers库（如SD3）或多模态生成模型[魔乐社区](https://modelers.cn/models/)推理工程仓（如Open-Sora-Plan v1.2、Flux.1-dev、HunyuanVideo），请先确保依据推理工程仓可以正常进行浮点推理。
- Open-Sora-Plan v1.2推理工程仓：[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)
- Flux.1-dev推理工程仓：[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)
- HunyuanVideo推理工程仓[MindIE/hunyuan_video](https://modelers.cn/models/MindIE/hunyuan_video)


### SD3-Medium W8A8静态量化

当前仅支持对SD3模型的transformer部分进行W8A8静态量化。

#### 量化启动脚本
校准数据Dump和量化的示例代码如下：

```python
# 导入模型库
import os
import torch
from diffusers import StableDiffusion3Pipeline

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, f'calib_data_{get_rank()}.pth')
safe_tensor_path = os.path.join(SAFE_TENSOR_FOLDER, f'rank_{get_rank()}.safetensors')


############################ 加载模型 ############################
def load_t2v_checkpoint(model_path):
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16).to('npu')
    return pipeline


pipeline = load_t2v_checkpoint("/path/to/stable-diffusion-3-medium-diffusers")  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    
    pipe(
        prompts=["A photo of an astronaut riding a horse on mars"],
        negative_prompts=[""],
        width=args.width,
        height=args.height,
        num_inference_steps=args.infer_steps,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据，校准数据需要提前dump生成
calib_dataset = torch.load(dump_data_path, map_location=f'npu:{os.getenv("RANK", 0)}')

# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "w8a8": W8A8ProcessorConfig(
            cfg = W8A8QuantConfig(
                act_method='minmax'
            ),
            disable_names=['context_embedder']
        ),
        "save": SaveProcessorConfig(
            output_path=os.path.dirname(safe_tensor_path),
            safetensors_name=os.path.basename(safe_tensor_path),
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=calib_dataset,
    device='npu'
)

# python pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```


示例启动脚本如下(请提前确保calib_prompts.txt权限不大于'0o640')：
```shell
python /the/absolut/path/of/example/multimodal_sd/SD3/sd3_inference.py \
    --sd3_model_path "/path/to/stable-diffusion-3-medium-diffusers" \
    --prompt_path "example/multimodal_sd/SD3/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 28 \
    --seed 42 \
    --device "npu" \
    --save_path "./results/quant/images" \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8"
```

### Open-Sora-Plan v1.2 W8A8静态量化
Open-Sora-Plan v1.2的推理量化依赖于推理工程仓：[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)，根据该工程仓完成配置后，使用以下示例代码进行量化。

校准数据Dump和量化的示例代码如下：

```python
import os
import torch

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, f'calib_data_{get_rank()}.pth')
safe_tensor_path = os.path.join(SAFE_TENSOR_FOLDER, f'rank_{get_rank()}.safetensors')


############################ 加载模型 ############################
def load_t2v_checkpoint():
    pass


pipeline = load_t2v_checkpoint(model_path)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    run_model_and_save_images(
        pipeline,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据，校准数据需要提前dump生成
calib_dataset = torch.load(dump_data_path, map_location=f'npu:{os.getenv("RANK", 0)}')

# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "w8a8": W8A8ProcessorConfig(
            cfg=W8A8QuantConfig(
                act_method='minmax'
            ),
            disable_names=get_disable_layer_names(model, layer_include=None,
                                                    layer_exclude=('*net.2*', '*adaln_single*'))
        ),
        "save": SaveProcessorConfig(
            output_path=os.path.dirname(safe_tensor_path),
            safetensors_name=os.path.basename(safe_tensor_path),
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=calib_dataset,
    device='npu'
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)

```

#### 量化启动脚本

我们也提供了完整的量化启动脚本示例：[OpenSoraPlanV1_2/inference.py](OpenSoraPlanV1_2/inference.py)，其启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：
```shell
# 根据使用卡数进行配置多卡环境变量和nproc_per_node，以下使用8卡为例
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
export TASK_QUEUE_ENABLE=2
export HCCL_OP_EXPANSION_MODE="AIV"
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    /the/absolut/path/of/example/multimodal_sd/OpenSoraPlanV1_2/inference.py \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 93 \
    --height 720 \
    --width 1280 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt "example/multimodal_sd/OpenSoraPlanV1_2/calib_prompts.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --dtype bf16 \
    --use_cfg_parallel \
    --algorithm "dit_cache" \
    --save_img_path "./results/quant/images" \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8"
```

### FLUX 时间步量化

**注意**: 在模型pipeline的去噪循环中，需要在每个timestep开始时调用`TimestepManager.set_timestep_idx()`来设置当前的时间步。

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager

# 在去噪循环中设置timestep
for step_id, t in enumerate(timesteps):
    # ----------- w8a8_timestep quantization -----------
    TimestepManager.set_timestep_idx(step_id)  # 必须在每个timestep开始时调用
    # ----------- w8a8_timestep quantization -----------

    model_output = pipeline(...)
    ...
```
例如在FLUX.1-dev/FLUX1dev/pipeline/pipeline_flux.py的FluxPipeline类的__call__函数中，添加如下代码：
```python
with self.progress_bar(total=num_inference_steps) as progreess_bar:
    for i,t in enumerate(timesteps):
        if self.interrupt:
            continue
        # -----------新增代码-----------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
        TimestepManager.set_timestep_idx(i)
        # -----------新增代码-----------
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
```

#### 量化启动脚本

校准数据Dump和量化的示例代码如下：

```python
import os
import torch

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8TimeStepProcessorConfig, W8A8TimeStepQuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, f'calib_data_{get_rank()}.pth')
safe_tensor_path = os.path.join(SAFE_TENSOR_FOLDER, f'rank_{get_rank()}.safetensors')


############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='timestep')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=20,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = torch.load(dump_data_path, map_location=f'npu:{get_rank()}')

# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "w8a8_timestep": W8A8TimeStepProcessorConfig(
            cfg=W8A8TimeStepQuantConfig(
                act_method='minmax'
            ),
            disable_names=get_disable_layer_names(
                model,
                layer_include='*',
                layer_exclude='*net.2*',
                ),
            timestep_sep=25,

        ),
        "save": SaveProcessorConfig(
            output_path=os.path.dirname(safe_tensor_path),
            safetensors_name=os.path.basename(safe_tensor_path),
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=calib_dataset,
    device='npu'
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

示例的启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# do quant
python /the/absolut/path/of/example/multimodal_sd/Flux/inference_flux.py \
    --path ${model_path} \
    --save_path "./results/quant/img" \
    --device_id 0 \
    --device "npu" \
    --prompt_path "example/multimodal_sd/Flux/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 50 \
    --seed 42 \
    --use_cache \
    --device_type "A2-64g" \
    --batch_size 1 \
    --max_num_prompt 0 \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_timestep"
```


### Flux fa3 量化

校准数据Dump和量化的示例代码如下：

```python

import os
import torch

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import FA3ProcessorConfig, W8A8DynamicQuantConfig, W8A8DynamicProcessorConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, f'calib_data_{get_rank()}.pth')
safe_tensor_path = os.path.join(SAFE_TENSOR_FOLDER, f'rank_{get_rank()}.safetensors')


############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=20,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = torch.load(dump_data_path, map_location=f'npu:{get_rank()}')

# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
    "fa3": FA3ProcessorConfig(),
    "w8a8_dynamic": W8A8DynamicProcessorConfig(
        cfg=W8A8DynamicQuantConfig(
            act_method='minmax'
        ),
        disable_names=[],

    ),
    "save": SaveProcessorConfig(
        output_path=os.path.dirname(safe_tensor_path),
        safetensors_name=os.path.basename(safe_tensor_path),
        json_name=None,
        save_type=['safe_tensor'],
        part_file_size=None
    )
},
calib_data=calib_dataset,
device='npu'
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

#### FAQuantizer 插入位置

在FLUX.1-dev/FLUX1dev/layers/attention_processor.py文件中插入fa3量化，请备份attention_processor.py文件，后续进行浮点推理或其他方式量化时恢复attention_processor.py文件。
```python
# 1 实例化FAQuantizer类
@maybe_allow_in_graph
class Attention(nn.Module):
    def __init__(self,...) 
        ...
         
        if processor is None:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

        # --------------------fa3-----------------------------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
        from msmodelslim import logger 

        from types import SimpleNamespace

        if is_tp == False:
            sp_size = 1
        else:
            import torch.distributed as dist

            sp_size = dist.get_world_size()

        # 创建配置字典
        config_dict = {
            'num_attention_heads': self.heads // sp_size, 
            'hidden_size': self.inner_dim,
            'num_key_value_heads': self.heads // sp_size,
            }

        # 转换为 SimpleNamespace 对象
        config = SimpleNamespace(**config_dict)
        self.fa_quantizer = FAQuantizer(config, logger=logger)
        # --------------------fa3-----------------------------

# 2 通过调用FAQuantizer的quant函数，对Q、K、V矩阵进行量化
class FluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(self,...)
        ...
        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb_mindspeed(query, image_rotary_emb)
            key = apply_rotary_emb_mindspeed(key, image_rotary_emb)

        # --------------------fa3-----------------------------
        query = attn.fa_quantizer.quant(query, qkv="q")
        key = attn.fa_quantizer.quant(key, qkv="k")
        value = attn.fa_quantizer.quant(value, qkv="v")
        # --------------------fa3-----------------------------

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = apply_fa(query, key, value, attention_mask)
        hidden_states = hidden_states.to(query.dtype)
        B, S, H = hidden_states.shape

        ....

# 3 通过调用FAQuantizer的quant函数，对Q、K、V矩阵进行量化
class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        ...)

        ...

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb_mindspeed(query, image_rotary_emb)
            key = apply_rotary_emb_mindspeed(key, image_rotary_emb)

        # --------------------fa3-----------------------------
        query = attn.fa_quantizer.quant(query, qkv="q")
        key = attn.fa_quantizer.quant(key, qkv="k")
        value = attn.fa_quantizer.quant(value, qkv="v")
        # --------------------fa3-----------------------------        
    
        hidden_states = apply_fa(query, key, value, attention_mask)
        hidden_states = hidden_states.to(query.dtype)
        ....
```

#### 量化启动脚本

示例的启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# do quant
python /the/absolut/path/of/example/multimodal_sd/Flux/inference_flux.py \
    --path ${model_path} \
    --save_path "./results/quant/img" \
    --device_id 0 \
    --device "npu" \
    --prompt_path "example/multimodal_sd/Flux/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 50 \
    --seed 42 \
    --use_cache \
    --device_type "A2-64g" \
    --batch_size 1 \
    --max_num_prompt 0 \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_dynamic_fa3"
```

### Flux 异常值抑制量化

校准数据Dump和量化的示例代码如下：

```python

import os
import torch

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import M3ProcessorConfig, M4ProcessorConfig, M6ProcessorConfig, W8A8DynamicQuantConfig, \
    W8A8DynamicProcessorConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, f'calib_data_{get_rank()}.pth')
safe_tensor_path = os.path.join(SAFE_TENSOR_FOLDER, f'rank_{get_rank()}.safetensors')


############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=50,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = torch.load(dump_data_path, map_location=f'npu:{get_rank()}')

# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
    "m4": M4ProcessorConfig(),
    "w8a8_dynamic": W8A8DynamicProcessorConfig(
        cfg=W8A8DynamicQuantConfig(
            act_method='minmax'
        ),
        disable_names=get_disable_layer_names(
            model,
            layer_include='*',
            layer_exclude='*net.2*',
        ),

    ),
    "save": SaveProcessorConfig(
        output_path=os.path.dirname(safe_tensor_path),
        safetensors_name=os.path.basename(safe_tensor_path),
        json_name=None,
        save_type=['safe_tensor'],
        part_file_size=None
    )
},
calib_data=calib_dataset,
device='npu'
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

#### 量化启动脚本

示例的启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# do quant
python /the/absolut/path/of/example/multimodal_sd/Flux/inference_flux.py \
    --path ${model_path} \
    --save_path "./results/quant/img" \
    --device_id 0 \
    --device "npu" \
    --prompt_path "example/multimodal_sd/Flux/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 50 \
    --seed 42 \
    --use_cache \
    --device_type "A2-64g" \
    --batch_size 1 \
    --max_num_prompt 0 \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_dynamic" \
    --anti_method "m4"
```

### HunyuanVideo 时间步量化

**注意**: 在模型pipeline的去噪循环中，需要在每个timestep开始时调用`TimestepManager.set_timestep_idx()`来设置当前的时间步。

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager

# 在去噪循环中设置timestep
for step_id, t in enumerate(timesteps):
    # ----------- w8a8_timestep quantization -----------
    TimestepManager.set_timestep_idx(step_id)  # 必须在每个timestep开始时调用
    # ----------- w8a8_timestep quantization -----------

    model_output = pipeline(...)
    ...
```
例如在hunyuan_video/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py的HunyuanVideoPipeline类的__call__函数中，添加如下代码：
```python
with self.progress_bar(total=num_inference_steps) as progreess_bar:
    for i,t in enumerate(timesteps):
        if self.interrupt:
            continue
        # -----------新增代码-----------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
        TimestepManager.set_timestep_idx(i)
        # -----------新增代码-----------
        latent_model_input = (
            torch.cat([latents] * 2)
            if self.do_classifier_free_guidance
            else latents
        )
```

#### 量化启动脚本

校准数据Dump和量化的示例代码如下：

```python
import os
import torch

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8TimeStepProcessorConfig, W8A8TimeStepQuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, f'calib_data_{get_rank()}.pth')
safe_tensor_path = os.path.join(SAFE_TENSOR_FOLDER, f'rank_{get_rank()}.safetensors')


############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='timestep')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=20,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = torch.load(dump_data_path, map_location=f'npu:{get_rank()}')

# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "w8a8_timestep": W8A8TimeStepProcessorConfig(
            cfg=W8A8TimeStepQuantConfig(
                act_method='minmax'
            ),
            disable_names=get_disable_layer_names(
              model,
              layer_include=['*double_blocks*', '*single_blocks*'],
              layer_exclude=['*img_mod*', '*modulation*', '*fc2*'],
              ),
            timestep_sep=25,

        ),
        "save": SaveProcessorConfig(
            output_path=os.path.dirname(safe_tensor_path),
            safetensors_name=os.path.basename(safe_tensor_path),
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=calib_dataset,
    device='npu'
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
    quant_model(model, session_cfg)

```


示例的启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# 根据使用卡数进行配置多卡环境变量和nproc_per_node，以下使用8卡为例
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=0

torchrun --nproc_per_node=8 /the/absolut/path/of/example/multimodal_sd/HunYuanVideo/sample_video.py \
    --model-base HunyuanVideo \
    --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
    --text-encoder-path HunyuanVideo/text_encoder \
    --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
    --model-resolution "720p" \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "example/multimodal_sd/HunYuanVideo/calib_prompts.txt" \
    --seed 42 \
    --flow-reverse \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --vae-parallel \
    --num-videos 1 \
    --save-path ./results \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_timestep"
      
```

### HunyuanVideo fa3 量化

校准数据Dump和量化的示例代码如下：

```python

import os
import torch

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import FA3ProcessorConfig, W8A8DynamicQuantConfig, W8A8DynamicProcessorConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, f'calib_data_{get_rank()}.pth')
safe_tensor_path = os.path.join(SAFE_TENSOR_FOLDER, f'rank_{get_rank()}.safetensors')


############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=20,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = torch.load(dump_data_path, map_location=f'npu:{get_rank()}')

# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "fa3": FA3ProcessorConfig(), 
        "w8a8_dynamic": W8A8DynamicProcessorConfig(
            cfg = W8A8DynamicQuantConfig(
                act_method = 'minmax'
            ),
            disable_names=get_disable_layer_names(
                model, 
                layer_include=('*double_blocks*', '*single_blocks*'),
                layer_exclude=('*img_mod*', '*modulation*'),
            ),
        ),
        "save": SaveProcessorConfig(
            output_path=os.path.dirname(safe_tensor_path),
            safetensors_name=os.path.basename(safe_tensor_path),
            save_type=["safe_tensor"],
            json_name=None,
            part_file_size=None,
        )
    },
    calib_data=calib_dataset[:20],
    device = "npu",
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

#### FAQuantizer 插入位置

在hunyuan_video/hyvideo/modules/models.py文件中插入fa3量化，请备份models.py文件，后续进行浮点推理或其他方式量化时恢复models.py文件。
```python
# 在导入处添加
from hyvideo.utils.parallel_mgr import get_sequence_parallel_world_size

# 1
class MMDoubleStreamBlock(nn.Module):
    def __init__(self,...) 
        ...
         
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        # ATTENTION_CACHE PARAMETER
        self.cache = None

        # --------------------fa3-----------------------------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
        from msmodelslim import logger 
        from types import SimpleNamespace

        sp_size = get_sequence_parallel_world_size()

        # 创建配置字典
        config_dict = {
            'num_attention_heads': self.heads_num // sp_size, 
            'hidden_size': hidden_size // sp_size,
            'num_key_value_heads': self.heads_num // sp_size,
            }

        # 转换为 SimpleNamespace 对象
        config = SimpleNamespace(**config_dict)
        self.fa_quantizer = FAQuantizer(config, logger=logger)
        # --------------------fa3-----------------------------

    ...    
    def double_forward(
        self,...)
        
        ...

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        
        # --------------------fa3-----------------------------
        q = self.fa_quantizer.quant(q, qkv="q")
        k = self.fa_quantizer.quant(k, qkv="k")
        v = self.fa_quantizer.quant(v, qkv="v")
        # --------------------fa3-----------------------------

        # attention computation start
        if not self.hybrid_seq_parallel_attn:

# 2
class MMSingleStreamBlock(nn.Module):
    def __init__(self,...) 
        ...
         
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        # ATTENTION_CACHE PARAMETER
        self.cache = None

        # --------------------fa3-----------------------------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
        from msmodelslim import logger 
        from types import SimpleNamespace

        sp_size = get_sequence_parallel_world_size()

        # 创建配置字典
        config_dict = {
            'num_attention_heads': self.heads_num // sp_size, 
            'hidden_size': self.hidden_size // sp_size,
            'num_key_value_heads': self.heads_num // sp_size,
            }

        # 转换为 SimpleNamespace 对象
        config = SimpleNamespace(**config_dict)
        self.fa_quantizer = FAQuantizer(config, logger=logger)
        # --------------------fa3-----------------------------
    
    def enable_deterministic(self):
        self.deterministic = True

    ...    
    
    def single_forward(
        self,...)
        
        ...

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        
        # --------------------fa3-----------------------------
        q = self.fa_quantizer.quant(q, qkv="q")
        k = self.fa_quantizer.quant(k, qkv="k")
        v = self.fa_quantizer.quant(v, qkv="v")
        # --------------------fa3-----------------------------

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
```

#### 量化启动脚本

示例的启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# 根据使用卡数进行配置多卡环境变量和nproc_per_node，以下使用8卡为例
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=0
torchrun --nproc_per_node=8 /the/absolut/path/of/example/multimodal_sd/HunYuanVideo/sample_video.py \
    --model-base HunyuanVideo \
    --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
    --text-encoder-path HunyuanVideo/text_encoder \
    --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
    --model-resolution "720p" \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "example/multimodal_sd/HunYuanVideo/calib_prompts.txt" \
    --seed 42 \
    --flow-reverse \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --vae-parallel \
    --num-videos 1 \
    --save-path ./results \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_dynamic_fa3"
```

### HunyuanVideo 异常值抑制量化

校准数据Dump和量化的示例代码如下：

```python

import os
import torch

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import M3ProcessorConfig, M4ProcessorConfig, M6ProcessorConfig, W8A8DynamicQuantConfig, \
  W8A8DynamicProcessorConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, f'calib_data_{get_rank()}.pth')
safe_tensor_path = os.path.join(SAFE_TENSOR_FOLDER, f'rank_{get_rank()}.safetensors')


############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=50,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = torch.load(dump_data_path, map_location=f'npu:{get_rank()}')

# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "m4": M4ProcessorConfig(), 
        "w8a8_dynamic": W8A8DynamicProcessorConfig(
            cfg = W8A8DynamicQuantConfig(
                act_method = 'minmax'
            ),
            disable_names=get_disable_layer_names(
                model, 
                layer_include=['*double_blocks*', '*single_blocks*'],
                layer_exclude=['*img_mod*', '*modulation*', '*fc2*'],
            ),
        ),
        "save": SaveProcessorConfig(
            output_path=os.path.dirname(safe_tensor_path),
            safetensors_name=os.path.basename(safe_tensor_path),
            save_type=["safe_tensor"],
            json_name=None,
            part_file_size=None,
        )
    },
    calib_data=calib_dataset,
    device = "npu",
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

#### 量化启动脚本

示例的启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# 根据使用卡数进行配置多卡环境变量和nproc_per_node，以下使用8卡为例
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=0
torchrun --nproc_per_node=8 /the/absolut/path/of/example/multimodal_sd/HunYuanVideo/sample_video.py \
    --model-base HunyuanVideo \
    --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
    --text-encoder-path HunyuanVideo/text_encoder \
    --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
    --model-resolution "720p" \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "example/multimodal_sd/HunYuanVideo/calib_prompts.txt" \
    --seed 42 \
    --flow-reverse \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --vae-parallel \
    --num-videos 1 \
    --save-path ./results \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_dynamic" \
    --anti_method "m4"
```
