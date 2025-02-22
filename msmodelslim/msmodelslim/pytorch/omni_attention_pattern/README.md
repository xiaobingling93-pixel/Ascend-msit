# Omni attention pattern 搜寻工具

&ensp;本工具用于搜索attention pattern，采用遗传算法，可以定制化地适配不同的sparsity(0-100)要求：sparsity为0的时候是模型原型不作任何改变，sparsity为100的时候是Omni attention的极致性能，压缩所有层的KV值，但精度可能会随着sparsity的增加而下降。<br>
&ensp;同一模型，只需要对浮点模型做一次attention pattern搜索，可用于该模型的所有量化模型。<br>
&ensp;需要注意的是pattern的搜索很重要，一个好的pattern不仅能有效地降低推理成本，推理的精度也能得到保障，本工具已在Qwen2.5-7B、Qwen2.5-72B、Llama3-8B、Llama3-70B做了验证，并获得最佳pattern。

表1 已验证模型列表

| 模型名称         |框架|
|--------------|-----|
| Qwen2.5-7B   |PyTorch|
| Llama3-8B    |PyTorch|
| Qwen2.5-72B  |PyTorch|
| Llama3.1-70B |PyTorch|

### 前提条件
已参考环境准备，完成CANN开发环境的部署、PyTorch 2.1.0及以上版本的安装及Python环境变量的配置。
执行命令安装如下依赖。
```
pip3 install numpy
pip3 install transformers>=4.45.2
pip3 install torch>=2.1.0
pip3 install torch_npu>=2.1.0
```

### 接口说明
OmniAttentionConfig类：

| 参数名          | 含义            |是否必填| 使用说明                                                  |
|--------------|---------------|----|-------------------------------------------------------|
| pool_size    | 控制遗传算法初始化个体数量 |非必填| 数据类型为INT，默认值为50，建议值在50左右（搜寻时间在10h左右）。个体数量越大，耗时越久      |
| num_mutation | 每轮进化变异的个体数量   |非必填| 数据类型为INT，默认值为10，建议使用默认值。仅在调用search_incremental()方法中使用 |
| model_path   | 模型路径          |必填| 数据类型为string                                           |
| save_path    | 保存路径          |必填| 数据类型为string                                           |
| seed         | 随机种子          |非必填| 数据类型为INT，默认值为42                                       |
<br>
OmniAttentionGeneticSearcher类：

|参数名|含义|是否必填|使用说明|
|----|----|----|----|
|config|OmniAttentionConfig实例|必填|数据类型OmniAttentionConfig类|

search_incremental()
```python
# 执行遗传搜索算法，寻找最佳的注意力模式。
# 
# 输出：0-90稀疏度，步长为10，每个稀疏度对应的最佳pattern，以文件的形式保存
OmniAttentionGeneticSearcher.search_incremental()
```
search_on_this_sparsity(sparsity)
```python
# 以指定的稀疏度，寻找最佳的注意力模式。
# 
# 输入：sparsity，数据类型为INT，表示稀疏度，数值范围为0~100，超过此范围数值将会被截断
# 输出：指定稀疏度下最佳注意力模式，以文件的形式保存
OmniAttentionGeneticSearcher.search_on_this_sparsity(sparsity)
```
                                                                                                                                                                               
### 搜索步骤（以Qwen2.5-7B为例）

#### 步骤1：用户准备原始模型。

用户需要自行准备模型、权重文件。本样例以Qwen2.5-7B-Instruct为例，从HuggingFace下载权重文件，目录示例如下：
```
.gitattributes
README.md
chat_template.json
config.json
generation_config.json
merges.txt
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
model.safetensors.index.json
tokenizer_config.json
tokenizer.json
tokenizer_config.json
vocab.json
```

#### 步骤2：创建pattern脚本示例

```
from msmodelslim.pytorch.omni_attention_pattern.omni_config import OmniAttentionConfig
from msmodelslim.pytorch.omni_attention_pattern.omni_tools import OmniAttentionGeneticSearcher

config = OmniAttentionConfig(model_path="{步骤一创建的模型路径}", save_path="{保存路径}", pool_size=50)

searcher = OmniAttentionGeneticSearcher(config)
searcher.search_on_this_sparsity(sparsity=50)
```

参数`pool_size`控制遗传算法初始化个体的数量。参数`sparsity`控制得到的pattern的稀疏度，`sparsity`越大，则压缩力度越大，pattern中的压缩头的数量越多，也就是说，推理时的性能越快，对精度的影响也可能会更大。

#### 步骤3：检查输出
搜索出的最佳pattern会保存在用户指定的“save_path”文件夹下，每种模型有一个自己的子文件夹。例如:

```
- {save_path}/
-- Qwen2.5-7B-Instruct/                            #模型名称
---- genetic_rowwise_sparsity_20_score_80.tsv      #示例生成文件

生成命名规则：
search_incremental()的生成文件：
    genetic_rowwise_sparsity_{sparsity}_score_{score}.tsv
    其中sparsity为当前稀疏度，score为得分，即在规定数据集中正确的数量
    
search_on_this_sparsity(sparsity)的生成文件：
    genetic_rowwise_on_this_sparsity_{sparsity}_score_{score}.tsv
    其中，sparsity为输入稀疏度，score为得分，用户可直接选取分数最高（即为最后轮次）结果
```

#### 步骤4：使用pattern
在MindIE中，可以通过使用环境变量来开启OMNI Attention，例如：
```
export ATB_LLM_OMNI_ATTENTION_ENABLE=1
export ATB_LLM_OMNI_SHIFT_WINDOWS_ENABLE=1
export ATB_LLM_OMNI_ATTENTION_PATTERN_FILE={步骤三用户生成pattern路径，指定到具体tsv文件}
```
只要如上所示指定pattern对应的tsv文件，MindIE会读取该pattern并用于推理加速。