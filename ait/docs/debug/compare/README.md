# ait debug compare功能使用指南

## 简介
- compare一键式全流程精度比对（推理）功能将推理场景的精度比对做了自动化，适用于 TensorFlow、ONNX、Caffe 模型，用户只需要输入原始模型，对应的离线模型和输入，输出整网比对的结果，离线模型为通过 ATC 工具转换的 om 模型，输入 bin 文件需要符合模型的输入要求（支持模型多输入）。
- 该功能使用约束场景说明，参考链接：[CANN商用版/约束说明（仅推理场景）](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/atlasaccuracy_16_0035.html)
- 对于 Caffe 模型，目前不支持动态 shape 的模型比对。对于 `yolov2` / `yolov3` / `ssd` 等需要自定义实现层的模型，需要自行编译安装特定版本的 caffe。
- 大模型加速库在线推理精度比对，参考链接：[加速库精度比对介绍](../../../examples/cli/debug/compare/acl_cmp_introduction/introduction.md)
- **注意**：请确保ATC工具转换的om与当前运行环境使用的芯片型号一致。


## 工具安装
- 一般工具安装请见 [ait一体化工具使用指南](../../../README.md)
- 此外还提供容器安装方式（支持caffe精度比对）

### 容器方式安装
容器方式安装目前提供了Ubuntu 18.04的docker镜像。在`<ait_project_root_path>/ait/components/debug/compare`目录下运行以下命令以构建镜像：
```shell
docker build --build-arg CANN_TOOLKIT_PATH=Ascend-cann-tookit<version+arch>.run --build-arg CANN_AMCT_PATH=Ascend-cann-amctt<version+arch>.tar.gz \ 
--build-arg CAFFE_SRC=caffe-ascend-amct.zip -f Dockerfile . -t ait-caffe:latest
```
注意:
1. 非root用户请加上sudo
2. 请将Ascend-cann-tookit<version+arch>.run改为实际上的toolkit路径(必须是相对路径)
3. 从这个[仓库](https://github.com/lenLRX/caffe)下载zip[代码](https://github.com/lenLRX/caffe/archive/refs/heads/ascend-amct.zip),得到的zip包可能叫ascend-amct.zip或caffe-ascend-amct.zip
4. 从[这里](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)下载amct的包Ascend-cann-amct_5.1.RC1.1_linux-aarch64.tar.gz(注意下载对应需要的版本如：X86，aarch64等)
5. 执行命令构建docker镜像,要求:
   * CANN_AMCT_PATH=步骤4下载的amct包名字
   * CAFFE_SRC=步骤3下载的caffe代码zip包
   运行以下命令以上述镜像启动容器：
```shell
docker run -it -v=`pwd`:/work   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/bin/npu-smi:/usr/bin/npu-smi \
-v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons --device /dev/davinci0 --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm  ait-caffe:latest
```
在启动容器时将driver路径挂载到容器中，指定映射的device设备。
` -v=`pwd`:/work `为将当前目录映射到容器work目录下（非必须）。


## 使用方法
### 功能介绍
#### 使用入口
compare功能可以直接通过ait命令行形式启动精度对比。启动方式如下：

**不指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```

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
│   └-- {onnx or tf or caffe} # 原模型 dump 数据存放路径，onnx / tf / caffe 分别对应 ONNX / Tensorflow / Caffe 模型
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # 随机输入数据，若指定了输入数据，则该文件不存在
├-- model
│   ├-- {om_model_name}.json                    # 离线模型om模型(.om)通过atc工具转换后的json文件
│   └-- new_{onnx_model_name}.onnx              # 把每个算子作为输出节点后新生成的 onnx 模型
│   └-- custom_op_{onnx_model_name}.onnx        # 若指定了--custom-op，删除自定义算子后的onnx子图模型
│   └-- new_custom_op_{onnx_model_name}.onnx    # 若指定了--custom-op，删除自定义算子后的onnx子图模型，并把每个算子作为输出节点后新生成的 onnx 模型
├-- result_{timestamp}.csv                   # 比对结果文件
└-- tmp                                      # 如果 -m 模型为 Tensorflow pb 文件, tfdbg 相关的临时目录
```
#### 输出结果说明和分析步骤参考

请移步[对比结果分析步骤](../../../examples/cli/debug/compare/result_analyse/README.md)


#### 命令行入参说明

| 参数名          | 描述                                                                                                                                                                                                                                                                                            | 必选 |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----|
| -gm，--golden-model | 模型文件 [.pb, .onnx, .prototxt, .om] 路径，分别对应 TF, ONNX, Caffe, OM 模型                                                                                                                                                                                                                               | 是  |
| -om，--om-model | 昇腾AI处理器的离线模型 [.om, .mindir]                                                                                                                                                                                                                                                                             | 是  |
| -w，--weight  | -gm 为 Caffe 模型时对应的权重文件（.caffemodel）                                                                                                                                                                                                                                                           | 否  |
| -i，--input   | 模型的输入数据路径，默认根据模型的input随机生成，多个输入以逗号分隔，例如：/home/input\_0.bin,/home/input\_1.bin,/home/input\_2.npy。注意：使用aipp模型时该输入为om模型的输入,且支持自动将npy文件转为bin文件                                                                                                                                                   | 否  |
| -c，--cann-path | CANN包安装完后路径，默认会从从系统环境变量`ASCEND_TOOLKIT_HOME`中获取`CANN` 包路径，如果不存在则默认为 `/usr/local/Ascend/ascend-toolkit/latest`                                                                                                                                                                                 | 否  |
| -o，--output  | 输出文件路径，默认为当前路径                                                                                                                                                                                                                                                                                | 否  |
| -is，--input-shape | 模型输入的shape信息，默认为空，例如"input_name1:1,224,224,3;input_name2:3,300",节点中间使用英文分号隔开。input_name必须是转换前的网络模型中的节点名称                                                                                                                                                                                      | 否  |
| -d，--device  | 指定运行设备 [0,255]，可选参数，默认0                                                                                                                                                                                                                                                                       | 否  |
| --output-nodes | 用户指定的输出节点。多个节点用英文分号（;）隔开。例如:"node_name1:0;node_name2:1;node_name3:0"                                                                                                                                                                                                                          | 否  |
| --output-size | 指定模型的输出size，有几个输出，就设几个值，每个值默认为**90000000**，如果模型输出超出大小，请指定此参数以修正。动态shape场景下，获取模型的输出size可能为0，用户需根据输入的shape预估一个较合适的值去申请内存。多个输出size用英文分号（,）隔开, 例如"10000,10000,10000"                                                                                                                              | 否  |
| --advisor    | 在比对结束后，针对比对结果进行数据分析，给出专家建议                                                                                                                                                                                                                                                                    | 否  |
| -dr，--dym-shape-range | 动态Shape的阈值范围。如果设置该参数，那么将根据参数中所有的Shape列表进行依次推理和精度比对。(仅支持onnx模型)<br/>配置格式为："input_name1:1,3,200\~224,224-230;input_name2:1,300"。<br/>其中，input_name必须是转换前的网络模型中的节点名称；"\~"表示范围，a\~b\~c含义为[a: b :c]；"-"表示某一位的取值。 <br/>                                                                             | 否  |
| --dump       | 是否dump所有算子的输出并进行精度对比。默认是True，即开启全部算子输出的比对。(仅支持onnx模型)<br/>使用方式：--dump False                                                                                                                                                                                                                   | 否  |
| --convert    | 支持om比对结果文件数据格式由bin文件转为npy文件，生成的npy文件目录为./dump_data/npu/{时间戳_bin2npy} 文件夹。使用方式：--convert True                                                                                                                                                                                                  | 否  |
| -cp, --custom-op | 支持存在NPU自定义算子的模型进行精度比对，onnx模型中存在的NPU自定义算子类型名称，当前支持范围："DeformableConv2D"、"BatchMultiClassNMS"、"RoiExtractor"，使用方法：--custom-op="DeformableConv2D"，或者传入多个自定义算子类型：--custom-op="BatchMultiClassNMS,RoiExtractor"，中间使用英文分号隔开。                                                                        | 否  |
| --locat      | 开启后,自动在每次比对结束后,对误差超阈值的首个节点(任一类误差),执行误差定位流程,自动定位误差的区间范围(无论单节点还是累计误差)。使用方式：--locat True                                                                                                                                                                                                         | 否  |
| -ofs, --onnx-fusion-switch| onnxruntime算子融合开关，默认**开启**算子融合，如存在onnx dump数据中因算子融合导致缺失的，建议关闭此开关。使用方式：--onnx-fusion-switch False                                                                                                                                                                                              | 否  |
| -single, --single-op| 单算子比对模式，默认关闭，开启时在输出路径下会生成single op目录，存放单算子比对结果文件使用方式：-single True                                                                                                                                                                                                                             | 否  |
| --fusion-switch-file| 昇腾模型融合规则配置文件，传入该文件后，compare工具会根据传入的融合规则配置文件，重新生成一个om文件，和--om-model传入的模型进行精度比较，例如：--fusion-switch-file ./fusion_switch.cfg，其中fusion_switch.cfg文件配置方法参见：[如何关闭/开启融合规则](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/reference/graphubfusionref/graphubfusionref_000003.html) | 否  |
| -max, --max-cmp-size| 表示每个dump数据比较的最大字节数，用于精度比对过程提速，默认0(0表示全量比较)，当模型中算子的输出存在较大shape的、比较过于耗时情况，可以尝试打开。注意：需要使用最新cann版本(>=6.3.RC3)。使用方式：--max-cmp-size 1024                                                                                                                                                            | 否  |
| -q,--quant_fusion_rule_file| 量化算子映射关系文件（昇腾模型压缩输出的json文件）。仅推理场景支持本参数。使用方式：--quant_fusion_rule_file                                                                                                                                                                                                                          | 否  |
| aclcmp | 加速库精度比对子命令，大模型加速库在线推理场景下使用，使用指导请移步：[使用指导](../../../examples/cli/debug/compare/11_pta_acl_cmp/basic_usage.md)                                                                                                                                                                                  |  |

### 使用场景

请移步[compare使用示例](../../../examples/cli/debug/compare/)

| 使用示例                                                                                             | 使用场景                                 |
|--------------------------------------------------------------------------------------------------|--------------------------------------|
| [01_basic_usage](../../../examples/cli/debug/compare/01_basic_usage)                             | 基础示例，运行onnx和om模型精度比对                 |
| [02_specify_input_data](../../../examples/cli/debug/compare/02_specify_input_data)               | 指定模型输入数据                             |
| [03_save_output_data](../../../examples/cli/debug/compare/03_save_output_data)                   | 指定结果输出目录                             |
| [04_specify_input_shape_info](../../../examples/cli/debug/compare/04_specify_input_shape_info)   | 指定模型输入的shape信息(动态场景必须进行指定)。          |
| [05_aipp_model_compare](../../../examples/cli/debug/compare/05_aipp_model_compare)               | 提供模型转换开启aipp参数的om模型与onnx模型进行精度比对的功能。 |
| [06_npu_custom_op](../../../examples/cli/debug/compare/06_npu_custom_op)                         | onnx模型中存在NPU自定义算子场景                  |
| [07_caffe_model](../../../examples/cli/debug/compare/07_caffe_model)                             | 标杆模型为Caffe框架的一键式精度比对                 |
| [08_accuracy_error_location](../../../examples/cli/debug/compare/08_accuracy_error_location)     | 误差及累计误差一键式自动定位                       |
| [09_single_op](../../../examples/cli/debug/compare/09_single_op)                                 | 单算子比对模式                              |
| [10_fusion_switch_file](../../../examples/cli/debug/compare/10_fusion_switch_file)               | 关闭融合规则.om模型和原始.om模型精度比对              |
| [11_mixing_precison_compare](../../../examples/cli/debug/compare/11mixing_precison_compare)      | 混合精度策略的.om模型和.om模型的精度比对              |
| [acl_cmp_introduction](../../../examples/cli/debug/compare/acl_cmp_introduction/introduction.md) | 大模型加速库在线推理精度比对介绍                     |

### 常见问题FAQ

* [compare常见问题](./FAQ.md)