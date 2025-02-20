# OmniAttentionConfig
用于构建[OmniAttentionGeneticSearcher](OmniAttentionGeneticSearcher.md)的配置类

### 函数原型
```python
OmniAttentionConfig(pool_size=100, num_mutation=10, model_path=None, seed=42)
```

### 参数说明
|参数名|含义|是否必填| 使用说明                                                  |
|----|----|----|-------------------------------------------------------|
|pool_size|控制遗传算法初始化个体数量|非必填| 数据类型为INT，默认值为100，建议值在50左右（搜寻时间在10h左右）。个体数量越大，耗时越久     |
|num_mutation|每轮进化变异的个体数量|非必填| 数据类型为INT，默认值为10，建议使用默认值。仅在调用search_incremental()方法中使用 |
|model_path|模型路径|必填| 数据类型为string                                           |
|seed|随机种子|非必填| 数据类型为INT，默认值为42                                       |

### 调用示例
根据实际需求，在OmniAttentionConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.omni_attention_pattern.omni_config import OmniAttentionConfig
from msmodelslim.pytorch.omni_attention_pattern.omni_tools import OmniAttentionGeneticSearcher

config = OmniAttentionConfig(model_path="{步骤一创建的模型路径}", pool_size=50)

searcher = OmniAttentionGeneticSearcher(config)
searcher.search_on_this_sparsity(sparsity=50)
```