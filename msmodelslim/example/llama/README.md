# LLAMA 量化案例

## 模型介绍

- [LLaMA（Large Language Model Meta AI）](https://github.com/facebookresearch/llama/tree/llama_v1)和 [LLaMA2（Large Language Model Meta AI 2）](https://github.com/facebookresearch/llama)，是由 Meta AI 发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 此矩阵罗列了各LLaMa模型支持的特性

| 模型及参数量 | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅300I DUO支持） |
|-------------|----------|-----------|--------------|--------------------------|
| LLaMa-7B | × | × | × | × |
| LLaMa-13B | × | × | × | × |
| LLaMa-33B | × | × | × | √ |
| LLaMa-65B | × | √ | × | × |
| LLaMa2-7B | √ | × | × | √ |
| LLaMa2-13B | √ | × | × | √ |
| LLaMa2-70B | √ | √ | × | × |

- 此模型仓已适配的模型版本
  - [LLaMa系列](https://github.com/facebookresearch/llama/tree/llama_v1)
  - [LLaMa2系列](https://github.com/facebookresearch/llama/tree/v2)

## 环境配置

1. 设置CANN包的环境变量

  ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

2. 下载安装开源版本msModelSlim
- git clone下载本仓代码
- 运行安装脚本
  ```shell
    cd msmodelslim
    bash install.sh
  ```
## 量化权重生成

- 量化权重统一使用 [quantifier.py](../common/quantifier.py)脚本生成，以下提供LLaMa模型量化权重生成快速启动命令

### 量化参数配置
量化参数作为命令行参数传给 quantifier.py 脚本，下面是该命令行的基本模板：
  ```shell
  python3 quantifier.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file boolq.jsonl --w_bit {权重量化bit} --a_bit {激活值量化bit}  --disable_names {手动回退的量化层名称} --device_type {device类型 cpu or npu} --disable_level {自动回退等级} --anti_method {异常值抑制方法} --act_method {激活值量化方法} --tokenizer_args {分词器可选参数} --use_kvcache_quant {是否使用kvcache量化功能}
  ```
- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例
- 请将{浮点权重路径}和{W8A8量化权重路径}替换为用户实际路径
- 如精度太差，推荐回退量化敏感层，按照以下方式配置 --disable_names 参数, 并设置回退层数 --disable_level：
  ```shell
    get_down_proj_disable_name() {
        local num_layer=$1
        local disable_names=""
        for ((i=0; i<$num_layer; i++)); do
            disable_names="$disable_names model.layers.$i.mlp.down_proj"
        done
        echo "$disable_names"
    }
    disable_names=$(get_down_proj_disable_name 32)
  ```
#### W8A8
- LLaMa2-7B/13B推荐使用W8A8 + Antioulier（离群值抑制）量化
生成llama2-7b量化权重，无回退层，antioutlier使用m1算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在CPU上进行运算
  ```shell
  python3 quantifier.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file boolq.jsonl --device_type cpu --disable_level L0 --anti_method m1 --act_method 1 --tokenizer_args {"padding_side":"left","pad_token":"<unk>"} --use_kvcache_quant False
  ```


#### W8A16
- LLaMa-65B、LLaMa2-70B推荐使用以下量化配置
  ```shell
  python3 quantifier.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file= teacher_qualification.jsonl --w_bit 8 --a_bit 16 --act_method 3 --tokenizer_args {"padding_side":"left","pad_token":"<unk>"}
  ```

#### W8A8SC 稀疏量化
- LLaMa2 7B/13B推荐使用以下稀疏量化配置
  ```shell
  python3 quantifier.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file teacher_qualification.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --tokenizer_args {"padding_side":"left","pad_token":"<unk>"}
  ```
- LLaMa1 33B推荐使用以下稀疏量化配置
  ```shell
  python3 quantifier.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file boolq.jsonl --act_method 2 --do_smooth True --use_sigma True --is_lowbit True --co_sparse True --w_bit 4 --tokenizer_args {"padding_side":"left","pad_token":"<unk>"}
  ```