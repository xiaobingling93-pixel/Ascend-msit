# OmniAttentionGeneticSearcher
### 功能说明
用于搜索attention pattern，根据指定的稀疏度压缩KV值
### 函数原型
```python
OmniAttentionGeneticSearcher(config)
```

### 参数说明
|参数名|含义|是否必填|使用说明|
|----|----|----|----|
|config|OmniAttentionConfig实例|必填|数据类型OmniAttentionConfig类|

### API接口说明
#### search_incremental()
```python
# 执行遗传搜索算法，寻找最佳的注意力模式。可选接口
# 
# 输出：0-90稀疏度，步长为10，每个稀疏度对应的最佳pattern，以文件的形式保存
OmniAttentionGeneticSearcher.search_incremental()
```

#### search_on_this_sparsity(sparsity)
```python
# 以指定的稀疏度，寻找最佳的注意力模式。可选接口
# 
# 输入：sparsity，必选参数，数据类型为INT，表示稀疏度，数值范围为0~100，超过此范围数值将会被截断
# 输出：指定稀疏度下最佳注意力模式，以文件的形式保存
OmniAttentionGeneticSearcher.search_on_this_sparsity(sparsity)
```

### 调用示例
```python
from msmodelslim.pytorch.omni_attention_pattern.omni_config import OmniAttentionConfig
from msmodelslim.pytorch.omni_attention_pattern.omni_tools import OmniAttentionGeneticSearcher

config = OmniAttentionConfig(model_path="{步骤一创建的模型路径}", save_path="{保存路径}", pool_size=50)

searcher = OmniAttentionGeneticSearcher(config)
searcher.search_on_this_sparsity(sparsity=50)
```
调用说明请参考[readme](../../../../msmodelslim/pytorch/omni_attention_pattern/README.md)