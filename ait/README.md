#  AIT

## 目录
- [AIT](#ait)
  - [目录](#目录)
  - [介绍](#介绍)
    - [AIT各子功能介绍](#ait各子功能介绍)
  - [工具安装](#工具安装)
  - [工具使用](#工具使用)
    - [命令行格式说明](#命令行格式说明)
  - [参考](#参考)
    - [AIT资源](#ait资源)
    - [常见问题FAQ](#常见问题faq)
  - [许可证](#许可证)
  - [公网URL说明](#公网url说明)
  - [免责声明](#免责声明)

## 介绍
AIT(Ascend Inference Tools)作为昇腾统一推理工具，提供客户一体化开发工具，用于辅助用户进行模型迁移以及性能与精度的调试调优，当前包括benchmark、debug、transplt、analyze、llm等组件。

### 模型推理迁移全流程
![模型推理迁移全流程](/ait_flow.png)

### 大模型推理迁移全流程
![大模型推理迁移全流程](/ait-llm-flow.png)

### AIT各子功能介绍
| 任务类型                                | 子功能                           | 说明                                       |
|-------------------------------------|-------------------------------|------------------------------------------|
| [benchmark](/ait/docs/benchmark)         | -                             | 用来针对指定的推理模型运行推理程序，并能够测试推理模型的性能（包括吞吐率、时延） |
| debug(一站式调试)                        | [surgeon](/ait/docs/debug/surgeon) | 使能ONNX模型在昇腾芯片的优化，并提供基于ONNX的改图功能          |
| debug(一站式调试)                        | [compare](/ait/docs/debug/compare) | 提供自动化的推理场景精度比对，用来定位问题算子                  |
| [analyze](/ait/components/analyze)       | -                             | 提供其他平台模型迁移至昇腾平台的支持度分析功能                  |
| [transplt](/ait/components/transplt)     | -                             | 提供NV C++推理应用迁移分析以及昇腾API推荐功能              |
| [convert](/ait/components/convert)       | -                             | 提供推理模型转换功能                               |
| [profile](/ait/docs/profile)             | -                             | 提供profiling，提供整网详细的性能数据及相关信息             |
| [llm](/ait/docs/llm/README.md) | -                             | 提供加速库（atb）大模型推理调试工具，包括数据dump功能和数据比对功能    |
| [tensor-view](/ait/docs/tensor_view)     | -                             | 提供查看、切片、转置、保存tensor的接口                   |


## 工具安装
[一体化安装指导](/ait/docs/install/README.md)


## 工具使用

### 命令行格式说明

ait工具可通过ait可执行文件方式启动，若安装工具时未提示Python的PATH变量问题，或手动将Python安装可执行文件的目录加入PATH变量，则可以直接使用如下命令格式：

```bash
ait <TASK> <SUB_TASK> [OPT] [ARGS]
```


其中，```<TASK>```为任务类型，当前支持debug、benchmark、transplt、analyze、convert、profile，后续可能会新增其他任务类型，可以通过如下方式```查看当前支持的任务列表```：

```bash
ait -h
```

```<SUB_TASK>```为子任务类型，当前在debug任务下面，有surgeon、compare，当前在profile任务下面，有msprof;
当前benchmark、analyze、convert、transplt任务没有子任务类型。后续其他任务会涉及扩展子任务类型，可以通过如下方式查看每个任务支持的```子功能列表```：

```bash
ait debug -h
```


```[OPT]```和```[ARGS]```为可选项以及参数，每个任务下面的可选项和参数都不同，以```debug任务下面的compare子任务```为例，可以通过如下方式```获取可选项和参数```


```bash
ait debug compare -h
```

## 参考

### AIT资源

* [AIT benchmark 快速入门指南](/ait/docs/benchmark/README.md)
* [AIT debug surgeon 快速入门指南](/ait/docs/debug/surgeon/README.md)
* [AIT debug compare 快速入门指南](/ait/docs/debug/compare/README.md)
* [AIT analyze 快速入门指南](/ait/components/analyze/README.md)
* [AIT transplt 快速入门指南](/ait/components/transplt/README.md)
* [AIT convert 快速入门指南](/ait/components/convert/README.md)
* [AIT profile 快速入门指南](/ait/docs/profile/README.md)
* [AIT llm 快速入门指南](/ait/components/llm/)

### 常见问题FAQ

* [AIT使用以及安装常见问题](https://gitee.com/ascend/ait/wikis/Home)
* [AIT安全拦截报错解决](https://gitee.com/ascend/ait/wikis/ait_security_error_log_solution)

## 许可证

[Apache License 2.0](/LICENSE)

## 公网URL说明
[公网URL说明](/ait/公网URL使用说明.csv)

## 免责声明

ait仅提供在昇腾设备上的一体化开发工具，支持一站式调试调优，不对其质量或维护负责。
如果您遇到了问题，Gitee/Ascend/ait提交issue，我们将根据您的issue跟踪解决。
衷心感谢您对我们社区的理解和贡献。

