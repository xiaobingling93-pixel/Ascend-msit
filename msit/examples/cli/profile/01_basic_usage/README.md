# Basic Usage


## 介绍
- 一键式全流程推理工具ait集成了Profiling性能分析工具，用于分析运行在昇腾AI处理器上的APP工程各个运行阶段的关键性能瓶颈并提出针对性能优化的建议，最终实现产品的极致性能。
- Profiling数据通过二进制可执行文件”msprof”进行数据采集，使用该方式采集Profiling数据需确保应用工程或算子工程所在运行环境已安装Toolkit组件包。
- 该工具使用约束场景说明，参考链接：[CANN商用版/约束说明（仅推理场景）](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/atlasaccuracy_16_0035.html)


## 运行示例
- **数据准备**
  - 昇腾AI处理器的离线模型（.om）路径
- **不指定模型输入** 命令示例，**其中--application中的路径需使用绝对路径**
  ```sh
  ait profile --application "ait benchmark -om /home/HwHiAiUser/resnet101_bs1.om" --output output_data/
  ```
  - `--application` 配置为运行环境上app可执行文件，可配置ait自带的benchmark推理程序，需配置ait自带的benchmark推理程序，具体使用方法参照参数说明及benchmark使用。
  - `-o, --output` (可选) 搜集到的profiling数据的存放路径，默认为当前路径下输出output目录
- **分析结果输出的目录结构**
```bash
|--- output_data/
|    |--- profiler/  # 采集的性能数据
|    |    |--- PROF_000001_20230608201922856_LPKNFOADMAQRMDGC/ # msprof保存的数据
|    |    |    |--- host/ # host侧数据
|    |    |    |--- device_0/ #device侧数据，调用多个device推理会有多个和device文件夹
|    |    |    |    |--- timeline/
|    |    |    |    |--- summary/
|    |    |    |    |--- data/
```
  - 输出的性能数据文件`timeline/`、`summary/`和`data/`中数据的含义参见[profile使用示例](../)中的其他使用示例

