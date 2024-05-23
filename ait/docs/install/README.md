# AIT 工具安装

> 以下为最新的安装方式。7.0.0 版本之前安装是通过shell脚本安装，可以参考《[安装指南（历史版本）](./history.md)》

## 环境和依赖

ait推理工具的安装包括**ait包**和**依赖的组件包**的安装，其中依赖包可以根据需求只添加所需要的组件包。


|   依赖软件名称   | 是否必选 | 版本 |                      备注                      |
|-----------------|---------|------|------------------------------------------------|
| CANN              | 必 选  | 建议安装CANN商业版6.3.RC1以上版本 | 请参见《[CANN-6.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/envdeployment/instg/instg_000002.html)》安装昇腾设备开发或运行环境，即toolkit软件包。安装后请根据安装提示配置环境变量（可以参考  [配置环境变量](#说明) ）。                       |
| GCC               | 必 选 | 7.3.0版本                                | 请参见《[GCC安装指引](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/envdeployment/instg/instg_000091.html)》安装GCC编译器（centos 7.6平台默认为gcc 4.8编译器，可能**无法安装**本工具，建议更新gcc编译器后再安装） |
| Python               | 必选 | 支持Python3.7.5+、Python3.8.x、Python3.9.x、Python3.10.x | 如使用TensorFlow模型的精度对比功能则需要Python3.7.5版本                                                                                                               |
| TensorFlow  | 非必选 | -                                      | 参考 [Centos7.6上TensorFlow1.15.0 环境安装](https://bbs.huaweicloud.com/blogs/181055) 安装 TensorFlow1.15.0 环境。(如不使用TensorFlow模型的精度对比功能则不需要安装)                                                  |
| Caffe    | 非必选 | -    | 参考 [Caffe Installation](http://caffe.berkeleyvision.org/installation.html) 安装 Caffe 环境。(如不使用 Caffe 模型的精度对比功能则不需要安装)                                                                    |
| Clang      | 非必选 | -    | 依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。(如不使用transplt应用迁移分析功能则不需要安装)                                                                                                     |


## AIT安装

安装方式包括：**源代码安装**和**pip源安装**，用户可以按需选取。
- [源代码安装](#源代码安装): 使用源码安装，保证是最新的 ait 功能。
- [pip源安装](#pip源安装): pip 安装 ms-ait 包，一般一个季度发包一次。

常见报错可以参照[Ait 安装常见问题](#常见问题-qa)

### 安装前置说明：
- 安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [CANN-6.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/envdeployment/instg/instg_000002.html)。
- 安装后用户可通过 **设置CANN_PATH环境变量** ，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest。
- 若不设置，工具默认会从环境变量ASCEND_TOOLKIT_HOME和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。


### 源代码安装

```shell
git clone https://gitee.com/ascend/ait.git
# 1. git pull origin 更新最新代码 
cd ait/ait

# 2. 安装 ait 包
pip install .

# 3. 通过ait install 命令，安装所需组件，例如需要安装 surgeon 和 llm 时：
ait install surgeon llm

# 4. 如果需要安装所有组件，可以使用 install all：
ait install all

# 5. 安装之后可以使用 ait check 命令检查安装是否成功：
ait check all 
```


### pip源安装

```shell
# 1. 安装 ait 包
pip install ms-ait

# 2. 通过ait install 命令，安装所需组件，例如需要安装 surgeon 和 llm 时：
ait install surgeon llm

# 3. 如果需要安装所有组件，可以使用 install all：
ait install all

# 4. 安装之后可以使用 ait check 命令检查安装是否成功：
ait check all 
```

> windows 下，仅支持安装 transplt 和 surgeon 组件

## 卸载

使用pip 卸载命令进行卸载。
1. 建议可以先看下目前安装的 ait 组件有哪些
```bash
# linux 
 pip list | grep -E "acl|ait|ais"
```
```shell
# windows
pip list | findstr ait
```
结果参考：
```log
(base) PS C:\workspce\ait> pip list | findstr ait
ait-surgeon             7.0.0rc2
ms-ait                  7.0.0rc2
```
2. 使用pip命令卸载您需要卸载的组件
```shell
# 仅仅卸载一个组件
pip uninstall ait-surgeon
# 卸载所有
pip uninstall ait-surgeon ms-ait
```
> 如果您仅仅卸载某几个组件，请不要将 ms-ait 卸载，会影响其他组件使用

# 离线安装(仅支持linux)
因为部分用户可能需要在非联网机器中安装 ait， 下面提供离线安装指导：

### 下载所有软件包
该步骤需要在可以联网的机器中(需要保证两台机器是同样的系统，平台，python版本)，首先需要安装 ait ,然后使用 ait download 命令下载所依赖的包：

```bash
# 1. 首先安装 ait, 使用源码方式
git clone https://gitee.com/ascend/ait.git
cd ait/ait
pip install .

# 2. 再使用 ait download 命令，下载对应组件到目录：
# 2.1 仅仅下载某几个组件，例如下载 llm 到 ./pkg-cache 目录:
ait download llm --dest ./pkg-cache 
# 2.2 安装所有组件
ait download all --dest ./pkg-cache
```

### 离线机器安装
该步骤，首先将上一步骤中相关文件拷贝到离线机器，包括：ait 源码，下载的依赖包。开始安装

```bash 
# 1. 首先安装 ait
cd ait/ait
pip install .

# 2. 开始安装组件，如 llm :
ait install llm --find-links=./pkg-cache

# 3. 安装之后可以使用 ait check 命令检查安装是否成功：
ait check all 
```

> 注意：
> 1. 离线安装不支持 windows 中安装
> 2. 离线安装首先要下载安装包，要保证下载安装包机器与离线安装机器一致。包括python版本，平台，操作系统等
> 3. 离线安装不支持 llvm 的安装，llvm 是 transplt 依赖的组件，请用户自行安装


# 安装相关命令行参数

## 安装命令： ait install 

| 参数名                         | 描述                                                         | 必选 |
| ------------------------------ | ------------------------------------------------------------ | ---- |
| comp_names                  | 是一个位置参数，指定需要安装的组件，当前可以选择的有：`llm`/`compare`/`surgeon`/`analyze`/`transplt`/`convert`/`msprof`/`benchmark`/`all`，具体选项可以使用`ait install --help` 查看。但指定为`all` 时，表明是需要安装所有组件 | 是   |
| --find-links/-f             | 查找包的路径，一般用于离线安装时 | 否  |
| --help                  | 帮助信息 | 否  |

```bash 
ait install llm
```

## 安装检查命令：ait check

| 参数名                         | 描述                                                         | 必选 |
| ------------------------------ | ------------------------------------------------------------ | ---- |
| comp_names                  | 是一个位置参数，指定需要检查的组件，当前可以选择的有：`llm`/`compare`/`surgeon`/`analyze`/`transplt`/`convert`/`msprof`/`benchmark`/`all`，具体选项可以使用`ait check --help` 查看。但指定为`all` 时，表明是需要检查所有组件 | 是   |
| --help                  | 帮助信息 | 否  |

```bash 
ait check all
```

## 附加构建命令：ait build-extra

部分组件在安装之后，会需要一些额外的构建动作，该步骤在 ait install 会自动执行。但是因为部分原因，比如前置包没有安装等，有可能构建失败。安装后可以使用 ait check 命令检查，会给出提示。用户可以使用 ait build-extra 重新构建。

| 参数名                         | 描述                                                         | 必选 |
| ------------------------------ | ------------------------------------------------------------ | ---- |
| comp_names                  | 是一个位置参数，指定需要构建的组件，当前可以选择的有：`llm`/`compare`/`surgeon`/`analyze`/`transplt`/`convert`/`msprof`/`benchmark`，具体选项可以使用`ait build-extra --help` 查看 | 是   |
| --find-links/-f             | 查找包的路径，一般用于离线安装时 | 否  |
| --help                  | 帮助信息 | 否  |

```bash 
ait build-extra llm
```

## 下载命令：ait download

下载安装包，用于离线安装场景

| 参数名                         | 描述                                                         | 必选 |
| ------------------------------ | ------------------------------------------------------------ | ---- |
| comp_names                  | 是一个位置参数，指定需要下载的组件，当前可以选择的有：`llm`/`compare`/`surgeon`/`analyze`/`transplt`/`convert`/`msprof`/`benchmark`/`all`，具体选项可以使用`ait download --help` 查看 | 是   |
| --dest                  | 目标路径 | 是   |
| --help                  | 帮助信息 | 否  |

```bash 
ait download llm
```

# 常见问题 Q&A

[参考：Ait 安装常见问题](./FAQ.md ':include')

