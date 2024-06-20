# Save Profiler or Dump Data


## 0 基本介绍
- 当开启可选命令`--profiler`，benchmark会调用msprof采集推理中的性能数据。
- 当开启可选命令`--dump`，benchmark推理过程中会保留全部算子的输出。
- 当开启`--acl-json-path`，可以在json文件中自定义profiler或dump的配置参数（只能选择其一）。

## 1 基本运行示例
### 1.1 --profiler 采集推理中的性能数据
- 示例命令：
```bash
ait benchmark --om-model /home/model/resnet50_v1.om --output ./output --profiler 1

```
- 输出的文件目录示例：
```bash
|--- output/
|    |--- 2023_06_08_19_27_summary.json # 汇总推理结果（推理总体性能数据）
|    |--- 2023_06_08_19_27/ # 输入文件
|    |    |--- pure_infer_data_0.bin
|    |--- profiler/  # 采集的性能数据
|    |    |--- PROF_000001_20230608201922856_LPKNFOADMAQRMDGC/ # msprof保存的数据
|    |    |    |--- host/ # host侧数据
|    |    |    |--- device_0/ #device侧数据，调用多个device推理会有多个和device文件夹

```
### 1.2 --dump 采集推理中的每层算子的输出数据
- 示例命令：
```bash
ait benchmark --om-model /home/model/resnet50_v1.om --output ./output --dump 1
```
- 输出的文件目录示例：
```bash
|--- output/
|    |--- acl.json # 与 --acl-json-path 命令配置的json文件相同
|    |--- 2023_06_08_19_27_summary.json  # 汇总推理结果（推理总体性能数据）
|    |--- 2023_06_08_19_27/
|    |    |--- pure_infer_data_0.bin # 输入文件
|    |--- dump/  # 采集的每层算子的输出数据
|    |    |--- 20230608192722/ # dump数据
|    |    |    |--- 0/
|    |    |    |    |--- resnet50_v1/
|    |    |    |    |    |--- 1/
|    |    |    |    |    |--- 0/
```

### 1.3 --acl-json-path 自定义采集推理中的数据
+ acl-json-path参数指定acl.json文件，可以在该文件中对应的profiler或dump参数。示例json文件如下：

  + 通过profiler采集推理中的性能数据

    ```bash
    # acl.json
    {
    "profiler": {
                  "switch": "on",
                  "output": "./result/profiler"
                }
    }
    ```
    更多性能参数配置请依据CANN包种类（商用版或社区版）分别参见《[CANN 商用版：开发工具指南/性能数据采集（acl.json配置文件方式）](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasprofiling_16_0086.html)》和《[CANN 社区版：开发工具指南/性能数据采集（acl.json配置文件方式）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasprofiling_16_0058.html)》中的参数配置详细描述

  + 通过dump采集算子的输出

    ```bash
    # acl.json
    {
        "dump": {
            "dump_list": [
                {
                    "model_name": "{model_name}"
                }
            ],
            "dump_mode": "output",
            "dump_path": "./result/dump"
        }
    }
    ```

    更多dump配置请参见《[CANN 开发工具指南/准备离线模型dump数据文件](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasaccuracy_16_0030.html)》中的“精度比对工具>比对数据准备>推理场景数据准备>准备离线模型dump数据文件”章节。

- 通过该方式进行profiler采集时，如果配置了环境变量`export AIT_NO_MSPROF_MODE=1`，输出的性能数据文件需要参见《[CANN 开发工具指南/数据解析与导出/Profiling数据导出](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasprofiling_16_0100.html)》，将性能数据解析并导出为可视化的timeline和summary文件。
- 通过该方式进行profiler采集时，如果**没有**配置环境变量`AIT_NO_MSPROF_MODE=1`，benchmark会将acl.json中与profiler相关的参数解析成msprof命令，调用msprof采集性能数据，结果默认带有可视化的timeline和summary文件，msprof输出的文件含义参考[性能数据采集（msprof命令行方式）](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasprofiling_16_0040.html)。
- 如果acl.json文件中同时配置了profiler和dump参数，需要要配置环境变量`export AIT_NO_MSPROF_MODE=1`保证同时采集

示例命令：
  ```bash
  ait benchmark --om-model ./resnet50_v1_bs1_fp32.om --acl-json-path ./acl.json
  ```
输出文件对应参考1.2与1.3节的示例。

## 2 拓展使用说明
### 2.1 --profiler 的自定义使用
+ profiler为固化到程序中的一组性能数据采集配置，生成的性能数据保存在--output参数指定的目录下的profiler文件夹内。

    该参数是通过调用ait/profiler/benchmark/infer/benchmark_process.py中的msprof_run_profiling函数来拉起msprof命令进行性能数据采集的。若需要修改性能数据采集参数，可根据实际情况修改msprof_run_profiling函数中的msprof_cmd参数。示例如下：

    ```bash
    msprof_cmd="{} --output={}/profiler --application=\"{}\" --model-execution=on --sys-hardware-mem=on --sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on".format(
            msprof_bin, args.output, cmd)
    ```
    该方式进行性能数据采集时，首先检查是否存在msprof命令：

    - 若命令存在，则使用该命令进行性能数据采集、解析并导出为可视化的timeline和summary文件。
    - 若命令不存在，则msprof层面会报错，benchmark层面不检查命令内容合法性。
    - 若环境配置了AIT_NO_MSPROF_MODE=1，则使用--profiler参数采集性能数据时调用的是benchmark构造的默认acl.json文件。

- msprof命令不存在或环境配置了AIT_NO_MSPROF_MODE=1情况下，采集的性能数据文件未自动解析。参考《[CANN 开发工具指南/数据解析与导出/Profiling数据导出](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasprofiling_16_0100.html)》，将性能数据解析并导出为可视化的timeline和summary文件。
- 更多性能数据采集参数介绍请参见《[CANN 开发工具指南/性能数据采集（msprof命令行方式）](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasprofiling_16_0041.html)》。

### 2.2 `--profiler` `--dump` 和 `--acl-json-path` 混合使用说明
  + acl-json-path优先级高于profiler和dump，同时设置时以acl-json-path为准。
  + profiler参数和dump参数，必须要增加output参数，指示输出路径。
  + profiler和dump可以分别使用，但不能同时启用。

## FAQ
使用出现问题时，可参考[FAQ](https://gitee.com/ascend/ait/wikis/benchmark_FAQ/ait%20benchmark%20%E5%AE%89%E8%A3%85%E9%97%AE%E9%A2%98FAQ)
