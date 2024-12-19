## ModelAdapter

### 功能说明
mindspeed的模型适配器，将mindspeed-LLM模型转化为modelslim可以量化的LLM模型。

### 函数原型
```python
ModelAdapter(model:nn.Module, dev_type='npu', forward_step=None, prefix='model.')
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 模型。| 必选。<br>数据类型：Mindspeed-LLM模型。 |
| dev_type | 输入 | 设备的device类型。| 可选。<br>数据类型：string。<br>可选值：['npu', 'gpu'], 默认值为'npu' |
| forward_step | 输入 | 启动Mindspeed-LLM模型的函数对象，当无法正常拉起模型时可以自定义一个启动器。| 可选。<br>数据类型：object。<br>默认值为None，会使用自带的模型拉起方式|
| prefix | 输入 | 模型state_dict的前缀名。| 可选。<br>数据类型：string。<br>可选值：["model.", "model.module."]<br>当模型的state_dict返回的keys和模型本身结构不一致时，必须通过调整prefix来保证一致性, 否则会影响模型保存的正确执行。|

### 调用示例
```python
from msmodelslim.pytorch.mindspeed import ModelAdapter
class GenerateForward:
    def __call__(self, model, x):
        return generate(model, x, tokens_to_generate=1)
model = ModelAdapter(model, forward_step=GenerateForward(), prefix='model.')
```
