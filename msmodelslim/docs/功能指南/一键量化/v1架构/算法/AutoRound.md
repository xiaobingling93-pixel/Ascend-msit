# AutoRound：低比特量化算法说明

## 背景和作用

- **来源**：Intel 开发的一种基于 SignSGD 的低比特量化方法。
- **背景**：传统量化方法（如四舍五入）在权重量化中并非最优选择，往往会引入较大的量化误差，从而显著降低模型精度，尤其在低比特（如 4bit 及以下）量化场景中表现更为明显。
- **核心思想**：通过引入可学习的舍入偏移参数，结合SignSGD优化器自适应调整各权重的舍入方向，并利用温度调度策略逐步硬化舍入操作，有效降低量化重构误差，在超低比特条件下实现模型精度与压缩效率的最优平衡。

## 使用方式

### 作为Processor使用：

```python
，# AutoRound支持混合量化，即对不同的层使用不同的量化配置，这里以 W8A8 和 W4A4 混合量化为例
# W8A8 动态量化配置
default_w8a8_dynamic: &default_w8a8_dynamic
  weight:
    scope: "per_group"        # 权重量化范围
    dtype: "int8"             # 权重量化数据类型
    symmetric: true           # 是否启用对称量化
    method: "autoround"       # 权重量化方法：AutoRound算法，即包含参数训练的权重量化
    ext:
      group_size: 256         # 量化组大小
      scale_dtype: "bfloat16" # 缩放因子数据类型
  act:
    scope: "per_token"        # 激活值量化范围
    dtype: "int8"             # 激活值量化数据类型
    symmetric: true           # 是否启用对称量化
    method: "minmax"          # 激活值量化方法：MinMax算法
    ext:
      group_size: -1          # 组大小：-1表示不使用分组，当前激活量化仅支持per-token
      dynamic: true           # 是否启用动态量化
      scale_dtype: "bfloat16" # 缩放因子数据类型

# W4A4 动态量化配置模板  
default_w4a4_dynamic: &default_w4a4_dynamic
  weight:
    scope: "per_group"
    dtype: "int4"
    symmetric: true
    method: "autoround"
    ext:
      group_size: 256
      scale_dtype: "bfloat16"
  act:
    scope: "per_token"
    dtype: "int4"
    symmetric: true 
    method: "minmax"
    ext:
      group_size: -1
      dynamic: true
      scale_dtype: "bfloat16"


spec:
  process:
    - type: "autoround_quant"     # 固定为 `autoround_quant`，用于指定 Processor 类型。
      iters: 400                  # 优化迭代次数
      enable_minmax_tuning: True  # 是否启用最小最大值调优
      enable_round_tuning: True   # 是否启用舍入调优
      strategies:
          # 策略1：除 up_proj、gate_proj 和 o_proj 层外，其余层均应用 W8A8 量化。
          - qconfig: *default_w8a8_dynamic
            exclude: 
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"
          # 策略2：对up_proj、gate_proj、o_proj层使用W4A4量化
          - qconfig: *default_w4a4_dynamic
            include: 
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"

```

### 参数说明

| 字段名               | 作用                   | 数据类型 | 默认值 | 说明                                                                                                  |
| -------------------- | ---------------------- | -------- | ------ | ----------------------------------------------------------------------------------------------------- |
| type                 | 处理器类型标识         | string   | -      | 固定值"autoround_quant"，用于标识该对象为 AutoRound 处理器。                                         |
| iters                | 优化迭代次数           | int      | 10     | 迭代次数，必须大于0。                                                                                 |
| enable_minmax_tuning | 是否启用最小最大值调优 | bool     | True   | 是否启用最小最大值调优，True表示启用，False表示不启用。                                               |
| enable_round_tuning  | 是否启用舍入调优       | bool     | True   | 是否启用舍入调优，True表示启用，False表示不启用。                                                     |
| strategies           | 量化策略               | list     | -      | 用于指定量化策略，支持int4和int8混合量化策略，量化粒度支持权重per_group和per-channel、激活per_token。 |

## 原理与实现

AutoRound 的核心思想是优化权重的舍入过程。该过程不是采用简单的四舍五入方式，而是基于 SignSGD（符号梯度下降）算法，自适应地学习每个权重的最佳舍入方向（向上或向下），并有针对性地调整缩放因子和零点。

### 核心公式

在传统量化中，权重 W 的量化公式通常为：

Ŵ = s × clip(⌊W/s + zp⌉, n, m)

其中 s 是缩放因子，zp 是零点，n 和 m 是量化后的上下界。

AutoRound 在此基础之上引入了可学习的舍入偏移 V 和可选的缩放因子调整参数 α 和 β：

Ŵ = s × clip(⌊W/s + zp + V⌉, n, m)，s = (max(W) × α - min(W) × β) / (2^bit - 1)

其中 V 用于控制舍入的方向，α 和 β 用于调整缩放因子的范围。

### 算法流程

该算法是一种逐层优化算法，对每一个decoder layer执行以下优化步骤：

1. **基准建立**：进行浮点前向传播，记录原始输出作为精度基准
2. **参数初始化**：对缩放因子、舍入偏移量进行初始化，并将其设置为可训练参数引入量化过程中
3. **量化重构**：执行量化-反量化操作，进行前向传播得到量化结果
4. **损失计算**：对比量化输出与浮点输出的差异，计算重构损失
5. **参数更新**：通过SignSGD优化器更新缩放因子和舍入偏移量
6. **迭代优化**：重复步骤1-5直至满足收敛条件或达到最大迭代次数
7. **最终量化**：应用迭代训练后的最优参数得到最终量化权重

### 实现

- 算法在 `msmodelslim/quant/processor/quant/autoround.py` 中实现，核心类为 `AutoroundQuantProcessor`：
  1. **初始化阶段**：

     * 层配置初始化：读取量化配置，并为每个网络层分配对应的量化配置方案
     * 参数预分配：初始化浮点输出、量化输出和最佳参数
  2. **pre_run阶段**：

     - 梯度冻结：关闭所有网络层的自动梯度计算，防止在训练过程中直接优化权重
  3. **preprocess阶段**：（逐层循环执行）

     * 基准输出采集：执行当前层浮点前向传播，并记录浮点输出结果作为优化基准
     * 线性层封装：对每个线性层进行包装处理，注入可训练的缩放因子和舍入偏移参数
     * 计算图构建：建立包含量化和反量化操作的可微计算图，支持梯度反向传播
  4. **process阶段**：（逐层循环执行）

     * 训练器初始化：设置学习率、迭代次数等参数，配置SignSGD优化器
     * 输入输出配置：使用上一层量化输出作为当前层输入，以浮点输出和量化输出的差距作为优化目标
     * 参数优化：通过多次迭代更新缩放因子和舍入偏移，最小化重构误差
     * 收敛监控：实时监测损失变化，达到收敛阈值或最大迭代次数时停止优化，得到最优参数
  5. **postprocess阶段**：（逐层循环执行）

     * 参数应用：将优化后的量化参数应用于对应层的权重量化
     * 解除封装：移除线性层的包装，恢复原始网络结构
     * 前向传播：执行当前层量化后的前向传播，作为下一层的输入

## 模型适配

### 接口与数据结构

```python
# 量化策略配置类
class QuantStrategyConfig(BaseModel):
    qconfig: LinearQConfig = Field(description="量化配置")
    include: List[str] = Field(default_factory=lambda: ["*"], description="包含的模块名称")
    exclude: List[str] = Field(default_factory=list, description="排除的模块名称")

# AutoroundProcess处理器配置
class AutoroundProcessorConfig(AutoProcessorConfig):
    type: Literal["autoround_quant"] = Field(default="autoround_quant", description="处理器类型标识")
    iters: int = Field(default=10, gt=0, description="迭代次数，必须大于0")
    enable_minmax_tuning: bool = Field(default=True, description="是否启用最小最大值调优")
    enable_round_tuning: bool = Field(default=True, description="是否启用舍入调优")
    strategies: List[QuantStrategyConfig] = Field(default_factory=list, description="量化策略配置列表")

# Autoround处理器
class AutoroundQuantProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter): ...

    def pre_run(self): ...
  
    def preprocess(self, request): ...

    def process(self, request): ...
  
    def postprocess(self, request): ...

    def post_run(self): ...

```

### 适配步骤

- **前置要求**：

  - 模型结构：支持线性层（2D权重张量）。
  - 量化配置：需明确指定数据类型、量化范围、对称性、量化粒度等参数。
  - 硬件环境：优化过程需要额外的内存来存储中间变量和梯度信息，支持Atlas A3 训练系列产品、Atlas A2 训练系列产品以及Atalas 800I A2推理产品。
- **步骤**：

  1. 在配置文件中定义量化策略，支持针对不同的层使用不同的量化策略。
  2. 在配置文件中使用"autoround_quant"指定autoround处理器，并且配置相关参数。
  3. 如需使用自定义校准集，可参考 `msmodelslim/lab_calib`添加数据集，并在配置文件中指定数据集名称。

### 适用范围和局限性

- **低比特量化**：适合极低比特量化场景中的4比特量化。
- **高精度需求**：在低比特条件下仍能保持较高的模型精度。
- **计算资源**：需要额外的优化过程，计算成本高于简单量化方法。
- **使用限制**：
  - 适用于llm中的线性层量化。
  - 需要足够的校准数据或训练迭代次数来优化参数。
  - **用于低比特量化时，建议用户配合Quarot或IterSmooth等异常值抑制方法一起使用，不建议用户（尤其是缺乏量化调优经验的基础用户）单独使用Autoround。因单独使用导致的模型精度严重下降或出现乱码等问题将无法保证支持，相关风险需自行承担。**

## 常见问题排查

### 1. 优化不收敛

- **现象**：在优化过程中，浮点结果与量化结果之间的差距波动较大或不收敛。
- **解决方案**：调整学习率或增加迭代次数。

### 2. 精度下降明显

- **现象**：量化后模型精度下降超过预期。
- **解决方案**：增加优化步数，调整量化配置，减少使用w4a4量化的层数，或使用更多更优质的校准数据。
