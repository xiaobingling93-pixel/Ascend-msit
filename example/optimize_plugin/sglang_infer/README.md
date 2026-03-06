# SGLang 插件化适配自动寻优工具使用说明

**说明：当前适配代码仅为插件化适配样例，仅做功能展示说明使用。详细代码适配指导可参考[自定义插件开发指导](https://gitcode.com/Ascend/msserviceprofiler/blob/master/docs/zh/serviceparam_optimizer_plugin_instruct.md)**

## 使用前准备

- SGLang服务化能正常拉起，能正常使用[ais_bench工具](https://gitee.com/aisbench/benchmark/blob/master/README.md)发送请求。
- 正常安装服务化自动寻优工具，具体安装方法可参考[服务化自动寻优工具](../../../msserviceprofiler/docs/serviceparam_optimizer_instruct.md)。

## 插件安装

- **源码安装**

```shell
git clone https://gitcode.com/Ascend/msit.git
cd msit/example/optimize_plugin/sglang_infer
pip install -e .
```

## 使用示例

 1. 修改自动寻优配置文件 msit/msserviceprofiler/msserviceprofiler/modelevalstate/config.toml。
    a. 新增sglang相关配置。

    ```shell
    # -------------------------sglang相关配置------------------------------------------------
    [sglang]
    [sglang.command]
    port = "30000"
    model = "your/model/path"
    device = "npu"
    others = ""
    [[sglang.target_field]]
    name = "MEM_FRACTION_STATIC"
    config_position = "env"
    min = 0.1
    max = 0.9
    dtype = "float"
    value = 0.8
    [[sglang.target_field]]
    name = "MAX_RUNNING_REQUESTS"
    config_position = "env"
    min = 2
    max = 50000
    dtype = "int"
    value = 1000
    [[sglang.target_field]]
    name = "MAX_QUEUED_REQUESTS" 
    config_position = "env"
    min = 1
    max = 10000
    dtype = "int"
    value = 200
    [[sglang.target_field]]
    name = "MAX_PREFILL_TOKENS" 
    config_position = "env"
    min = 1000
    max = 20000
    dtype = "int"
    value = 16000
    ```

    - - b. 修改ais_bench相关配置。

    ```shell
    # -------------------------测评工具相关配置------------------------------------------------
    [ais_bench.command]
    models = "vllm_api_stream_chat"
    datasets = "synthetic_gen"
    mode = "perf"
    num_prompts = 10
    ```

 2. 执行命令行。

  ```shell
  msserviceprofiler optimizer -e sgl_infer
  ```

**命令行参数，及config.toml配置文件说明请参考[服务化自动寻优工具](../../../msserviceprofiler/docs/serviceparam_optimizer_instruct.md)。**

## 结果说明

**寻优结果说明请参考[服务化自动寻优工具](../../../msserviceprofiler/docs/serviceparam_optimizer_instruct.md)。**
