# msServiceProfiler 工具介绍

## 简介

msServiceProfiler 是一款基于昇腾平台，支持MindIE框架和vLLM框架的服务化调优工具。

其性能采集与数据解析能力已嵌入昇腾CANN工具包，支持MindStudio Insight、Chrome Tracing、Grafana多个平台数据可视化。

目前，扩展能力**服务化性能数据比对工具**和**vLLM服务化性能采集工具**已在本仓库开源。

## 使用前准备
### 环境准备
- 硬件环境请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。

- 软件环境请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)》安装昇腾设备开发或运行环境，即toolkit软件包。

以上环境依赖请根据实际环境选择适配的版本。

### 安装

目前工具支持 pip 安装后直接调用，以及源码下载后脚本调用两种方式。
- **pip 安装 msserviceprofiler**
```shell
pip install -U msserviceprofiler
```

- **源码安装**
```shell
git clone https://gitcode.com/Ascend/msit.git
export PYTHONPATH=$PWD/msit/msserviceprofiler/:$PYTHONPATH
cd msit/msserviceprofiler
pip install -e .
```

## 功能介绍

- **服务化性能数据比对工具**

支持对使用msServiceProfiler工具采集的性能数据进行差异比对，通过比对快速识别可能存在的问题点。具体请参见[服务化性能数据比对工具](docs/服务化性能数据比对工具.md)。

- **vLLM服务化性能采集工具**

基于Ascend-vLLM，提供性能数据采集能力，结合msServiceProfiler的数据解析与可视化能力，可以vLLM服务化推理调试调优。具体请参见[vLLM服务化性能采集工具](docs/vLLM服务化性能采集工具.md)。

- **服务化自动寻优工具**

基于msServiceProfiler工具采集的性能数据，提供服务化参数自动寻优能力，可以对服务化的参数以及测试工具的参数进行寻优。具体请参见[服务化自动寻优工具](docs/serviceparam_optimizer_instruct.md)。

- **服务化专家建议工具**

基于benchmark 输出结果以及 service 的 config.json 配置，提供分析提高 TTFT / Throughput 等的优化点能力。具体请参见[服务化专家建议工具](docs/service_profiling_advisor_instruct.md)。

- **服务化多维度解析工具**

基于msServiceProfiler工具采集的性能数据，提供性能数据多维度分析能力，可以对性能数据进行batch维度、request维度和service维度分析。具体请参见[服务化多维度解析工具](docs/msServiceProfiler_multi_analyze_instruct.md)。

- **服务化拆解工具**

基于msServiceProfiler工具采集的性能数据，提供性能数据拆解能力，可以对batch内各阶段耗时进行分析。具体请参见[服务化拆解工具](docs/service_performance_split_tool_instruct.md)。