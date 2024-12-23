## FAQuantizer

### 功能说明
运行量化算法，对Q（Query）、K（Key）、V（Value）进行量化。

### 函数原型
```python
FAQuantizer(config, logger)
```

### 参数说明
| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ---------- | ---- | -------- |
| config | 输入 | 预训练配置类实例，包含模型参数。 | 必选。<br>数据类型：PretrainedConfig。 |
| logger | 输入 | 日志记录器，用于记录日志信息。 | 必选。<br>数据类型：Logger。 |

### 调用示例

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 

self.fa_quantizer = FAQuantizer(self.config, logger)

query_states = self.fa_quantizer.quant(query_states, qkv="q")
key_states = self.fa_quantizer.quant(key_states, qkv="k")
value_states = self.fa_quantizer.quant(value_states, qkv="v")
```