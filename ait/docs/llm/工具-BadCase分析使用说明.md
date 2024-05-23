# 一键式筛选问题数据

在评估模型时，不同的数据集常常伴随着不同的评测指标。比如最常见的中文评测数据集 `C-Eval`，其评测指标则是最简单的 `accuracy`，即正确率。但是模型在不同场景下，处理的任务不同，我们期望的结果也不同。譬如模型在处理翻译任务时，我们则无法通过 `accuracy` 来评测其输出好坏，换言之，翻译没有标准答案，言之有理即可。又或是在和标杆模型比较时，我们发现自己的模型对于部分提示词的提问，回答开始胡言乱语。那么何为言之有理，何为胡言乱语，这是一个难以界定的词汇，但是我们仍然可以将其抽象成多种量化指标，如语义，词义等等不同角度来评测模型的输出结果。

`CaseFilter` 是 `ait llm` 工具在此需求背景下，孕育出来的一个接口。它集成了不同的评测指标，使用起来也特别轻巧方便。目前还处于萌芽阶段，欢迎大家积极提供改进意见，帮助它成长。

## 接口定义
- `class CaseFilter`
    
    对外接口。创建类实例，后续通过实例添加[评测指标](#评测指标)并执行。构造函数不接受任何参数。
    ```py
    case_filter = CaseFilter()
    ```

    - `add_metrics(**metrics)`
        
        类方法。只接受字典型参数，`key` 为[评测指标](#评测指标)，`value` 为自定义阈值，如果阈值为 `None`，则使用该指标默认阈值。可一次添加多个指标，也可以多次调用此方法添加指标。例如:
        ```py
        add_metrics(bleu=None, rouge=None)

        # Alternatively

        add_metrics(bleu=None)
        add_metrics(rouge=None)
        ```

    - `apply(ins, outs, refs, output_dir=None)`
       - `ins`: List of Prompts
       - `outs`: List of model outputs
       - `refs`: The references (ground truth or label) to compare with
       - `output_dir`: 结果输出地址，默认 `None` 为当前工作路径
        
        类方法。执行指标评测，并根据阈值筛出问题数据
        ```py
        case_filter.apply(["prompts"], ["answers"], ["ground truth"]) # output_dir is None by default
        ```

## 使用方法

```py
# 调用接口
from ait_llm import CaseFilter


# 创建实例
case_filter = CaseFilter()
# 添加指标
case_filter.add_metrics(accuracy=None)
# 执行
case_filter.apply(
    ["How is your day?"],
    ["Good."],
    ["I am 23 years old."]
)
```

## 参考结果

<center>

| model input      | model output | gold standard      | Accuracy |
| :---------:      | :----------: | :-----------:      | :------: |
| How is your day? | Good.        | I am 23 years old. | 0        |

</center>

## Appendix
以下给出不同评测指标及其分数指示意见，和参考数据集。后续会收集添加更多数据集和推荐评测指标

### 评测指标

<center>
<a id="评测指标"></a>
<caption>评测指标</caption>

| Metrics                | 说明 | Score Range | Recommend Dataset       | Suggestions                            |
| :-----:                | :--: | :---------: | :---------------:       | :---------:                            |
| accuracy               | 正确率 | 0 or 1      | CEval, BoolQ, MMLU      | 1 表示完全 match，0 表示完全不 match     |
| edit_distance          | 编辑距离 | [0, +∞)     | IWSLT                   | 0 表示完全 match，score 越大表示越不相似 |
| bleu-(1, 2, 3, 4)                | 机器翻译质量评估 | [0, 1]      | IWSLT                   | 1 表示完全match，0 表示完全不 match      |
| rouge-(1, 2, l)                 | 文本摘要评估 | [0, 1]      | SQuAD, gsm8K            | 1 表示完全match，0 表示完全不 match      |
| relative_distinct-(1, 2, 3, 4)  | 输出多样性评估 | (0, +∞) | SQuAD, gsm8K              | $\begin{cases} \text{模型输出多样性\textbf{弱于}标杆} & \text{if } x \in (0, 1) \\ \text{模型输出多样性\textbf{等于}标杆} & \text{if } x = 1 \\ \text{模型输出多样性\textbf{强于}标杆} & \text{if } x \in (1, \infty) \end{cases}$ |
| relative_abnormal      | 字符异常率评估 | (0, +∞) | SQuAD, gsm8K              | $\begin{cases} \text{模型输出多样性\textbf{弱于}标杆} & \text{if } x \in (0, 1) \\ \text{模型输出多样性\textbf{等于}标杆} & \text{if } x = 1 \\ \text{模型输出多样性\textbf{强于}标杆} & \text{if } x \in (1, \infty) \end{cases}$ |


</center>
其中 `指标-(1, 2, 3, 4)` 表示此指标的不同变种，数字越大代表比较的越全面，越宏观。数字越小，代表比较的越精细。分数执导意见同样有效。

### 参考数据集
<center>

| Dataset  | Full Name | Tasks | Answers | Reference |
| :-----:  | :-------: | :---: | :-----: | :-------: |
| C-Eval   | Chinese Evaluation | Question Answering | Multiple Choices | https://huggingface.co/datasets/ceval/ceval-exam |
| BoolQ    | Boolean Questions | Question Answering | True or False | https://huggingface.co/datasets/google/boolq |
| MMLU     | Massive Multitask Language Understanding | Question Answering | Multiple Choices | https://huggingface.co/datasets/cais/mmlu |
| SQuAD    | Stanford Question Answering Dataset | Summarization | {"text": [], "answer start": []} | https://huggingface.co/datasets/rajpurkar/squad_v2 |
| HumanEval | Human Evaluation | Code Generation | Code Snippet | https://huggingface.co/datasets/openai_humaneval |
| gsm8k    | Grade School Math 8K | Question Answering | Logical Steps | https://huggingface.co/datasets/gsm8k |
| IWSLT    | International Conference on Spoken Language Translation | Translation | Sentence in Different Languages | https://huggingface.co/datasets/iwslt2017 |
| VQA      | Vision Question Answering    | Image Captioning | Descriptions of Images | https://huggingface.co/datasets/lmms-lab/VQAv2 |

</center>
