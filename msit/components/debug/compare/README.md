# 一键式全流程精度比对（推理）
### 功能介绍
- 一键式全流程精度比对（推理）工具将推理场景的精度比对做了自动化，适用于 TensorFlow 和 ONNX 模型，用户只需要输入原始模型，对应的离线模型和输入，输出整网比对的结果，离线模型为通过 ATC 工具转换的 om 模型，输入 bin 文件需要符合模型的输入要求（支持模型多输入）。
- 大模型加速库在线推理精度比对，参考链接：[加速库精度比对介绍](../../../examples/cli/debug/compare/acl_cmp_introduction/introduction.md)
- 该工具使用约束场景说明，参考链接：[CANN商用版/约束说明（仅推理场景）](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/atlasaccuracy_16_0035.html)

### 环境准备
- 已安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [昇腾文档](https://www.hiascend.com/zh/document)
- 安装 `python3.7.5` 环境
- **安装benchmark工具**，安装参考文档：[ais_bench推理工具使用指南](https://gitee.com/ascend/msit/tree/master/msit/components/benchmark/README.md)
- **`ONNX` 相关 python 依赖包** `pip3 install onnxruntime onnx numpy`，若 pip 安装依赖失败，建议执行命令 `pip3 install --upgrade pip` 进行升级，避免因 pip 版本过低导致安装失败
- **`TensorFlow` 相关 python 依赖包**，参考 [Centos7.6上tensorflow1.15.0 环境安装](https://bbs.huaweicloud.com/blogs/181055) 安装 TensorFlow1.15.0 环境

### 使用方法
- 通过压缩包方式或 git 命令获取本项目
  ```sh
  git clone https://gitee.com/ascend/msit.git
  ```
- 进入 compare 目录
  ```sh
  cd msit/msit/components/debug/compare
  ```
- 配置 CANN 包相关环境变量，其中 `/usr/local/Ascend/ascend-toolkit` 需使用实际 CANN 包安装后路径
  ```sh
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
- **数据准备**
  - 昇腾AI处理器的离线模型（.om）路径
  - 模型文件（.pb或.onnx）路径
  - (可选) 模型的输入数据（.bin）路径
- **不指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  python3 main.py -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  - `-om, –offline-model-path` 指定昇腾AI处理器的离线模型（.om）路径
  - `-m, --model-path` 指定模型文件（.pb或.onnx）路径
  - `-c，–-cann-path` (可选) 指定 `CANN` 包安装完后路径，不指定路径默认会从系统环境变量`ASCEND_TOOLKIT_HOME`中获取`CANN` 包路径，如果不存在则默认为 `/usr/local/Ascend/ascend-toolkit/latest`
  - `-o, –output-path` (可选) 输出文件路径，默认为当前路径
- **指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  python3 main.py -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  - `-i，–input-path` 模型的输入数据路径，默认根据模型的 input 随机生成，多个输入以逗号分隔，例如：`/home/input_0.bin,/home/input_1.bin`，本场景会根据文件输入 size 和模型实际输入 size 自动进行组 Batch，但需保证数据除 batch size 以外的 shape 与模型输入一致

### 输出结果说明
```sh
{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape} 用来区分动态shape时不同的模型实际输入，静态shape时没有该层
├-- dump_data
│   ├-- npu                          # npu dump 数据目录
│   │   ├-- {timestamp}              # 模型所有npu dump的算子输出，dump为False情况下没有该目录
│   │   │   └-- 0                    # Device 设备 ID 号
│   │   │       └-- {om_model_name}  # 模型名称
│   │   │           └-- 1            # 模型 ID 号
│   │   │               ├-- 0        # 针对每个Task ID执行的次数维护一个序号，从0开始计数，该Task每dump一次数据，序号递增1
│   │   │               │   ├-- Add.8.5.1682067845380164
│   │   │               │   ├-- ...
│   │   │               │   └-- Transpose.4.1682148295048447
│   │   │               └-- 1
│   │   │                   ├-- Add.11.4.1682148323212422
│   │   │                   ├-- ...
│   │   │                   └-- Transpose.4.1682148327390978
│   │   ├-- {time_stamp}
│   │   │   ├-- input_0_0.bin
│   │   │   └-- input_0_0.npy
│   │   └-- {time_stamp}_summary.json
│   └-- {onnx or tf}                         # -m 模型为 .onnx 时，onnx dump 数据目录，Tensorflow 模型为 tf
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # 随机输入数据，若指定了输入数据，则该文件不存在
├-- model
│   ├-- {om_model_name}.json
│   └-- new_{om_model_name}.onnx             # 把每个算子作为输出节点后新生成的 onnx 模型
├-- result_{timestamp}.csv                   # 比对结果文件
└-- tmp                                      # 如果 -m 模型为 Tensorflow pb 文件, tfdbg 相关的临时目录
```

### 比对结果分析
- **比对结果** 在文件 `result_{timestamp}.csv` 中，比对结果的含义与基础精度比对工具完全相同，其中每个字段的含义可参考 [CANN商用版/比对步骤（推理场景）](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/atlasaccuracy_16_0039.html)
- **analyser 分析结果** 在调用结束后打印，在全部对比完成后，逐行分析数据，排除 nan 数据，输出各对比项中首个差距不在阈值范围内的算子。

  | 对比项目                  | 阈值   |
  | ------------------------- | ------ |
  | CosineSimilarity          | <0.99  |
  | RelativeEuclideanDistance | >0.05  |
  | KullbackLeiblerDivergence | >0.005 |
  | RootMeanSquareError       | >1.0   |
  | MeanRelativeError         | >1.0   |

  输出结果使用 markdown 表格显示
  ```sh
  2023-04-19 13:54:10(1005)-[INFO]Operators may lead to inaccuracy:

  |                   Monitor |  Value | Index | OpType | NPUDump | GroundTruth |
  |--------------------------:|-------:|------:|-------:|--------:|------------:|
  |          CosineSimilarity | 0.6722 |   214 |    Mul |   Mul_6 |       Mul_6 |
  | RelativeEuclideanDistance |      1 |   214 |    Mul |   Mul_6 |       Mul_6 |
  ```
  **Python API 调用示例**
  ```py
  from msquickcmp.net_compare import analyser
  _ = analyser.Analyser('result_2021211214657.csv')()
  # |                   Monitor |  Value | Index | OpType | NPUDump | GroundTruth |
  # |--------------------------:|-------:|------:|-------:|--------:|------------:|
  # |          CosineSimilarity | 0.6722 |   214 |    Mul |   Mul_6 |       Mul_6 |
  # | RelativeEuclideanDistance |      1 |   214 |    Mul |   Mul_6 |       Mul_6 |
  ```
  Strategy 目前支持两种
  - `FIRST_INVALID_OVERALL` 只要有一项不满足就输出该条数据，并结束遍历，默认方式
  - `FIRST_INVALID_EACH` 输出每一个评估项首个不满足的
  ```py
  from msquickcmp.net_compare import analyser
  _ = analyser.Analyser('result_2021211214657.csv')(strategy=analyser.STRATEGIES.FIRST_INVALID_EACH)
  # |                   Monitor |   Value | Index |        OpType |                       NPUDump | GroundTruth # |
  # |--------------------------:|--------:|------:|--------------:|------------------------------:|------------:|
  # |          CosineSimilarity |  0.6722 |   214 |           Mul |                         Mul_6 |       Mul_6 |
  # | RelativeEuclideanDistance |       1 |   214 |           Mul |                         Mul_6 |       Mul_6 |
  # |       RootMeanSquareError |   4.061 |   241 |      ArgMaxV2 | PartitionedCall_ArgMax_118... | ArgMax_1180 |
  # |         MeanRelativeError |    2.99 |   241 |      ArgMaxV2 | PartitionedCall_ArgMax_118... | ArgMax_1180 |
  # | KullbackLeiblerDivergence | 0.01125 |   316 | BatchMatMulV2 |                    MatMul_179 |  MatMul_179 |
  ```

### 参数说明

| 参数名                      | 描述                                       | 必选   |
| ------------------------ | ---------------------------------------- | ---- |
| -m，--model-path          | 模型文件（.pb或.onnx)路径，目前只支持pb模型与onnx模型       | 是    |
| -om，--offline-model-path | 昇腾AI处理器的离线模型（.om）                        | 是    |
| -i，--input-path          | 模型的输入数据路径，默认根据模型的input随机生成，多个输入以逗号分隔，例如：/home/input\_0.bin,/home/input\_1.bin | 否    |
| -c，--cann-path           | CANN包安装完后路径，默认会从系统环境变量`ASCEND_TOOLKIT_HOME`中获取`CANN` 包路径，如果不存在则默认为 `/usr/local/Ascend/ascend-toolkit/latest`| 否    |
| -o，--output-path         | 输出文件路径，默认为当前路径                           | 否    |
| -s，--input-shape         | 模型输入的shape信息，默认为空，例如input_name1:1,224,224,3;input_name2:3,300,节点中间使用英文分号隔开。input_name必须是转换前的网络模型中的节点名称 | 否    |
| -d，--device              | 指定运行设备 [0,255]，可选参数，默认0                  | 否    |
| --output-nodes           | 用户指定的输出节点。多个节点用英文分号（;）隔开。例如:node_name1:0;node_name2:1;node_name3:0 | 否    |
| --output-size            | 指定模型的输出size，有几个输出，就设几个值。动态shape场景下，获取模型的输出size可能为0，用户需根据输入的shape预估一个较合适的值去申请内存。多个输出size用英文分号（,）隔开, 例如"10000,10000,10000" | 否    |
| --advisor           | 在比对结束后，针对比对结果进行数据分析，给出专家建议 | 否    |
| -dr，--dymShape-range     | 动态Shape的阈值范围。如果设置该参数，那么将根据参数中所有的Shape列表进行依次推理和精度比对。(仅支持onnx模型)<br/>配置格式为：input_name1:1,3,200\~224,224-230;input_name2:1,300。<br/>其中，input_name必须是转换前的网络模型中的节点名称；"\~"表示范围，a\~b\~c含义为[a: b :c]；"-"表示某一位的取值。 <br/> | 否  |
| --dump                   | 是否dump所有算子的输出并进行精度对比。默认是True，即开启全部算子输出的比对。(仅支持onnx模型)<br/>使用方式：--dump False            | 否  |
| --convert                 | 支持om比对结果文件数据格式由bin文件转为npy文件，生成的npy文件目录为./dump_data/npu/{时间戳_bin2npy} 文件夹。使用方式：--convert True | 否    |

### 执行案例
- 获取 Tensorflow pb 原始模型 [AIPainting_v2.pb](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.pb)
- 获取 om 模型 [AIPainting_v2.om](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.om)
- **参考上述使用方法，执行命令运行，如果需要运行指定模型输入，可以先执行第二种用户不指定模型输入命令，用随机生成的bin文件作为输入**  
