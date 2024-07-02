# msit profile功能使用指南

### 简介
- 一键式全流程推理工具msit集成了profiling性能分析工具，用于分析运行在昇腾AI处理器上的APP工程各个运行阶段的关键性能瓶颈并提出针对性能优化的建议，最终实现产品的极致性能。
- profiling数据通过二进制可执行文件”msprof”进行数据采集，使用该方式采集profiling数据需确保应用工程或算子工程所在运行环境已安装Toolkit组件包。
- 该工具使用约束场景说明，参考链接：[CANN商用版/约束说明（仅推理场景）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha003/developmenttools/devtool/atlasprofiling_16_0004.html)

### 环境准备
- 已安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [昇腾文档](https://www.hiascend.com/zh/document)
- 安装 `python3.7.5` 环境
- **安装msit工具**，安装参考文档：[msit工具安装](https://gitee.com/ascend/msit/blob/master/msit/docs/install/README.md)

### 使用方法
- 通过压缩包方式或 git 命令获取本项目
  ```sh
  git clone https://gitee.com/ascend/msit.git
  ```
- 进入 msprof 目录
  ```sh
  cd msit/msit/components/profile/msprof
  ```
- 配置 CANN 包相关环境变量，其中 `/usr/local/Ascend/ascend-toolkit` 需使用实际 CANN 包安装后路径
  ```sh
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
- **数据准备**
  - 昇腾AI处理器的离线模型（.om）路径
- **不指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  msit profile --application "msit benchmark -om /home/HwHiAiUser/resnet101_bs1.om" --output  /home/HwHiAiUser/result
  ```
  - `--application` 配置为运行环境上app可执行文件，可配置msit自带的benchmark推理程序，需配置msit自带的benchmark推理程序，具体使用方法参照参数说明及benchmark使用。
  - `-o, --output` (可选) 搜集到的profiling数据的存放路径，默认为当前路径下输出output目录

### 参数说明

  | 参数名                    | 描述                                       | 必选   |
  | ------------------------ | ---------------------------------------- | ---- |
  | --application            | 配置为运行环境上app可执行文件，可配置msit自带的benchmark推理程序，application带参数输入，此时需要使用英文双引号将”application”的参数值括起来，例如--application "msit benchmark -om /home/HwHiAiUser/resnet50.om"，用户使用仅需修改指定om路径 | 是    |
  | -o, --output             | 搜集到的profiling数据的存放路径，默认为当前路径下输出output目录                                                                | 否    |
  | --model-execution        | 控制ge model execution性能数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --sys-hardware-mem       | 控制DDR，LLC的读写带宽数据采集开关，可选on或off，默认为on。 | 否    |
  | --sys-cpu-profiling      | CPU（AI CPU、Ctrl CPU、TS CPU）采集开关。可选on或off，默认值为off。                           | 否    |
  | --sys-profiling          | 系统CPU usage及System memory采集开关。可选on或off，默认值为off。 | 否    |
  | --sys-pid-profiling      | 进程的CPU usage及进程的memory采集开关。可选on或off，默认值为off。 | 否    |
  | --dvpp-profiling         | DVPP采集开关，可选on或off，默认值为on | 否    |
  | --runtime-api            | 控制runtime api性能数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --task-time              | 控制ts timeline数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --aicpu                  | aicpu开关，可选on或off，默认为on。 | 否  |
  | -h, --help               | 工具使用帮助信息。               | 否  |

  ### 使用场景
请移步[profile使用示例](../../../examples/cli/profile/)
  | 使用示例               | 使用场景                                 |
  |-----------------------| ---------------------------------------- |
  | [01_basic_usage](../../../examples/cli/profile/01_basic_usage)    | 基础示例，对benchmark推理om模型执行性能分析       |
  | [02_collect_ai_task_data](../../../examples/cli/profile/02_collect_ai_task_data) | 采集AI任务运行性能数据 |
  | [03_collect_ascend_ai_processor_data](../../../examples/cli/profile/03_collect_ascend_ai_processor_data) | 采集昇腾AI处理器系统数据 |

  ### 性能分析实践案例
应用msit profile 进行性能分析可以参考案例:[基于msit的性能调优案例](https://gitee.com/ascend/msit/wikis/%E6%A1%88%E4%BE%8B%E5%88%86%E4%BA%AB/%E5%9F%BA%E4%BA%8Emsit%E7%9A%84%E6%80%A7%E8%83%BD%E8%B0%83%E4%BC%98%E6%A1%88%E4%BE%8B)