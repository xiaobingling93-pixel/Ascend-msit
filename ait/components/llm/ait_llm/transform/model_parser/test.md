## 测试方法

在parser.py中，有一个model_to_json函数，接受两个参数，model和name，其中model是torch model，可以使用以下方法构造

* 从transformers.models包导入

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
```

* 使用transformers库从本地文件导入

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM("/data/chatglm3-6b")
```
上面代码中，引号包裹的是一个路径，可以从huggingface下载模型目录，如何使用该目录

* 使用transformers库从huggingface下载
  由于网络问题，此方法不太适用，建议使用上面的方法

## 验证

parser.py中的主要函数是build_model_tree，它会生成如下结构：

### root

首先，最外层必须要有repeat_count和repeat_block这两个字段，代表模型中重复块的数量和结构

### repeat_block

repeat_block中必须要有的是attention和mlp这两个字段，大概率要有的是input_layernorm和post_attention_layernorm这两个字段

#### attention

attention可能有如下几个field

| 字段        | 含义               | 可选值/类型                       |
|-----------|------------------|------------------------------|
| structure | 结构类型             | q-k-v-o-r、q-kv-o-r、w-o-r、w-o |
| q         | query            | Linear结构                     |
| k         | key              | Linear结构                     |
| v         | value            | Linear结构                     |
| kv        | key+value        | Linear结构                     |
| w         | query+key+value  | Linear结构                     |
| o         | output           | Linear结构                     |
| rope      | rotary-embedding | RotaryEmbedding结构            |

其中structure值的含义是，例如q-kv-o-r，表示attention中有q、kv、o、r这几个字段。

###### Linear结构
```rust
struct Linear {
    kind: "Linear",
    in_features: u64,
    out_features: u64,
    bias: bool
}
```

###### RotaryEmbedding
```rust
struct RotaryEmbedding {
    kind: "RotaryEmbedding",
    base: u64 | -1,
    dim: u64 | -1,
    max_position_embeddings: u64 | -1,
    max_seq_len_cached: u64 | -1
}
```

#### mlp
mlp中必定有两个field，ff和act，其中ff中有两个或三个Linear结构，act中一般只有一个kind字符，大概率是SiLU，如果是GELU，那么还会有一个approximate字段，为true或false

#### input_layernorm/post_attention_layernorm
layernorm可以为LayerNorm结构或RMSNorm结构，反映在kind字段中

###### LayerNorm
```rust
struct LayerNorm {
    kind: "LayerNorm",
    normalized_shape: str,
    eps: f64,
    element_affine: bool,
    bias: bool
}
```

###### RMSNorm
```rust
struct RMSNorm {  
    kind: "RMSNorm",
    eps: u64
}
```

### children
root中可能还有children字段，其中可能有一个lm_head，大概率是一个Linear，children字段中
可能还有children字段，其中可能有一个embed_tokens字段，是一个Embedding结构。

```rust
struct Embedding {
    kind: "Embedding",
    num_embeddings: u64,
    embedding_dim: u64,
    padding_idx: u64
}
```
