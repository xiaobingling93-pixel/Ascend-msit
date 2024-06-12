# LLM 迁移分析
## PyTorch transformers LLM 模型迁移生成 atb 浮点模型
- 由 PyTorch transformers LLM 模型迁移生成 atb 浮点模型，包括 model、layer 层代码，以及相应的 cpp 与 h 文件

### 准备
- 安装 ait
- 获取 [Gitee ascend/MindIE-LLM](https://gitee.com/ascend/MindIE-LLM) 源码
- 准备待迁移的 transformers 模型目录

### QWEN 13B 迁移示例
- **迁移生成 atb 模型代码**，`--source` 指定待迁移的 transformers 模型目录，生成迁移后 model 以及 layer 的 cpp 与 h 代码
  ```sh
  ait llm transform -s /data/qwen-14b-chat
  # Generated files: [
  #     qwenlmheadmodel/model/decoder_model.cpp,
  #     qwenlmheadmodel/model/decoder_model.h,
  #     qwenlmheadmodel/layer/decoder_model.cpp,
  #     qwenlmheadmodel/layer/decoder_model.h,
  # ]
  ```
- 将生成的代码放到 `MindIE-LLM` 模型目录下，**该路径基于不同的 MindIE 版本可能会不同**
  ```sh
  mv qwenlmheadmodel ~/MindIE-LLM/src/modeling/backend/atb_framework/models
  ```
- 重新编译 `MindIE-LLM`，**实际的编译路径与命令需要参照 MindIE 文档**
  ```sh
  cd ~/src/modeling/backend/atb_framework
  bash scripts/build.sh
  ```
  由于迁移的适配性问题，以及 `MindIE-LLM` 迭代更新，编译过程可能存在报错，仍依赖用户手动修复错误
***

## llm 浮点模型 layer 层稀疏量化迁移
- 由 llm 的浮点模型 layer 层代码，迁移生成稀疏量化 layer 层代码，包括 cpp 文件与 h 文件

### 准备
- 安装 ait
- 获取 [Gitee ascend/MindIE-LLM](https://gitee.com/ascend/MindIE-LLM) 源码，找到待迁移模型 layer 定义

### Baichuan2 7B 迁移示例
- 模型 layer 定义位置
  ```sh
  cd ModelLink/mindie_ref/mindie_llm/atb_models/models/baichuan2/7b  # layer 定义位置
  ls
  # layer  model  operation
  ```
- 迁移为量化模型
  ```sh
  ait llm transform -s layer/flash_attention_rope_layer.cpp
  # ...
  # Transformed source files: [
  #     layer/flash_attention_rope_layer.cpp
  #     layer/flash_attention_rope_layer.h
  # ]
  # Transformed target files: [
  #     layer/quant_flash_attention_rope_layer.cpp
  #     layer/quant_flash_attention_rope_layer.h
  # ]
  ```
  - 由 `layer/flash_attention_rope_layer.cpp` `layer/flash_attention_rope_layer.h` 迁移生成 `layer/quant_flash_attention_rope_layer.cpp` `layer/quant_flash_attention_rope_layer.h`
  - 其中 `FlashAttentionRopeLayerTensorId` 节点中增加了 `_DESCALE` `_BIAS` 相关节点，同时 `RmsNorm` `Linear` `MLP` 等相关节点更新了属性及输入 tensor 节点
- 指定 `--enable-sparse` 迁移为稀疏量化模型
  ```sh
  ait llm transform -s layer/flash_attention_rope_layer.cpp --enable-sparse
  # ...
  # Transformed source files: [
  #     layer/flash_attention_rope_layer.cpp
  #     layer/flash_attention_rope_layer.h
  # ]
  # Transformed target files: [
  #     layer/sparse_quant_flash_attention_rope_layer.cpp
  #     layer/sparse_quant_flash_attention_rope_layer.h
  # ]
  ```
  - 由 `layer/flash_attention_rope_layer.cpp` `layer/flash_attention_rope_layer.h` 迁移生成 `layer/sparse_quant_flash_attention_rope_layer.cpp` `layer/sparse_quant_flash_attention_rope_layer.h`
  - 其中 `LinearParam` 更新为 `LinearSparseParam`，并在 `FlashAttentionRopeLayerTensorId` 增加了 `_INDEX` 相关节点

### 参数说明
```sh
ait llm transform -s {SOURCE} [--enable-sparse] [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

| 参数名          | 描述                                                                                                                             | 必选 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---- |
| -s, --source    | 指定待迁移的代码路径，支持指定文件夹，将迁移文件夹下所有不包含 `quant` 的文件；或指定单独的 cpp 文件，将迁移该 cpp 文件，以及同名的 h 文件 | 是   |
| --enable-sparse | 指定是否迁移为系数量化模型，不指定则迁移为量化模型                                                                                   | 否   |
| -l, --log-level | 指定log level，默认为 info，可选值 debug, info, warning, error, fatal, critical                                 | 否   |