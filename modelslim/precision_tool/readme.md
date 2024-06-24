# Precision Tool 说用方法说明
### 简单的例子
```python
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from precision_tool import PrecisionTest
import torch

if __name__ == '__main__':
    model_path = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True)
    tokenizer_params = {
        'revision': None,
        'use_fast': True,
        'padding_side': 'left',
        'truncation_side': 'left',
        'trust_remote_code': True
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_params)
    precision_test = PrecisionTest(model, tokenizer, "boolq", 1, "npu")
    precision_test.test()

```
### 接口介绍
#### 实例创建接口
```python
def __init__(self, model, tokenizer, dataset, batch_size, hardware_type,
             tokenizer_return_type_id=False):
    """
    @param model:
        llm to run the test, should be an instance of transformers.PreTrainedModel
    @param dataset:
        dataset to test precision
    @param batch_size:
        batch_size to run inference
    @param hardware_type:
        currently only npu is supported
    @param tokenizer_return_type_id:
        tokenizer return token type id
    """
```
其中
  + model: 待测试模型，当前需要为可采用 Transformers 库加载的模型
  + tokenizer: 与 model 配套的 tokenizer
  + dataset: 待测试数据集，当前支持 ceval_0_shot/ceval_5_shot/boolq/humaneval
  + hardware_type: 当前**仅**支持传入"npu"
  + tokenizer_return_type_id: 当输入 Bert 类型接口时需要传入 True，具体可以根据接口运行的反馈来确定
#### 测试结果接口
```python
def test(self):
```
### 使用方法
1. 下载数据集，并修改成如下的样式
|-- dataset
    |-- boolq
    |   `-- dev.jsonl
    |-- ceval_0_shot
    |   |-- val
    |       |-- Humanities
    |       |   `-- *.jsonl
    |       |-- Other
    |       |   `-- *.jsonl
    |       |-- STEM
    |       |   `-- *.jsonl
    |       `-- Social_Science
    |           `-- *.jsonl
    |-- ceval_5_shot
    |   |-- subject_mapping.json
    |   `-- val
    |       |-- accountant_val.csv
    |       |-- ...
    |       `-- veterinary_medicine_val.csv
    `-- humaneval
    `-- human-eval.jsonl
请保持文件夹名称与结构一致
2. 将该数据集放到与 precision_tool 同一个路径下
3. 如果测试 human-eval，则需要安装 https://github.com/openai/human-eval
注：
以当前时间 2024/05/21 为标杆时间，需要修改 https://github.com/openai/human-eval/blob/master/human_eval/execution.py#L58