### msModelSlim
### 简介

msModelSlim，即昇腾压缩加速工具，一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。支持训练加速和推理加速，包括模型低秩分解、稀疏训练、训练后量化、量化感知训练等功能，昇腾AI模型开发用户可以灵活调用Python API接口，对模型进行性能调优，并支持导出不同格式模型，在昇腾AI处理器上运行。

### 环境准备

- 使用msModelSlim工具前，需参考《CANN软件安装指南》搭建开发环境。
- 安装CANN软件后，需要以CANN运行用户登录环境，执行如下示例命令配置环境变量。

```
source {CANN包安装路径}/ascend-toolkit/set_env.sh
```
- 使用非root用户运行调优任务时，需要管理员用户将运行用户加入驱动运行用户组（例如：HwHiAiUser）中，保证普通用户对run包的lib库有读权限。
在线提单
- msModelSlim工具依赖Python，以Python3.7.5为例，请以运行用户执行如下命令设置Python3.7.5的相关环境变量。

```
#用于设置python3.7.5库文件路径
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
#如果用户环境存在多个python3版本，则指定使用python3.7.5版本
export PATH=/usr/local/python3.7.5/bin:$PATH
```


