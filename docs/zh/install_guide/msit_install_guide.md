
# **MindStudio Inferecen Tools 安装指南**

## 安装说明

MindStudio Inference Tools（MindStudio 昇腾推理工具链，msIT），为用户提供大模型与传统模型推理开发中常用的模型压缩、模型调试调优等功能，支持推理服务化场景下的性能调优能力，帮助用户达到最优的推理性能。本文主要介绍msit工具链的安装方法。

## 安装前准备

> [!Note] 说明
> 使用root安装会引入提权风险，本指南建议以普通用户安装使用

**下载安装包**

1. **访问OBS制品仓**：
    * 访问 openLiBing (OBS 制品仓)。
    * 在仓库中找到并单击 **"ascend-package"** 目录。
    * 进入该目录后，单击子目录 **"msit"**。

2. **下载对应架构的安装包**：
    msit 的安装包目前提供两种 CPU 架构版本，请根据您服务器的架构选择下载。
    * **x86_64 架构**：
        如果您使用的是基于 Intel/AMD 处理器的服务器，请下载此版本的安装包。[单击链接](https://ascend-package.obs.cn-north-4.myhuaweicloud.com/msit/Ascend-mindstudio-inference-toolkit_linux-x86_64.run)进行下载。
    * **aarch64 架构**：
        如果您使用的是基于华为鲲鹏（Kunpeng）等 ARM 处理器的服务器，请下载此版本的安装包。[单击链接](https://ascend-package.obs.cn-north-4.myhuaweicloud.com/msit/Ascend-mindstudio-inference-toolkit_linux-aarch64.run)进行下载。

3. **（可选）在终端通过 `wget` 下载**：
    您也可以在服务器终端中，使用 `wget` 命令直接下载对应的安装包。将以下链接替换为您需要的架构链接。

    ```bash
    # 下载 x86_64 架构安装包
    wget https://ascend-package.obs.cn-north-4.myhuaweicloud.com/msit/Ascend-mindstudio-inference-toolkit_linux-x86_64.run

    # 下载 aarch64 架构安装包
    wget https://ascend-package.obs.cn-north-4.myhuaweicloud.com/msit/Ascend-mindstudio-inference-toolkit_linux-aarch64.run
    ```

4. **将软件包保存至本地**：
    执行以上任何一步操作后，将对应的 `.run` 安装文件下载到您的本地或目标服务器目录。

**安装 CANN（可选）**

以下组件依赖CANN生态才能运行，如果您想使用如下工具，需要在安装msit之前先安装CANN：

- [**msProf（MindStudio Profiler）**](https://gitcode.com/Ascend/msprof)<br>
    **数据采集工具**：构建昇腾全场景性能调优基础能力，支持采集CANN和NPU性能数据，提升昇腾设备性能调优效率。

- [**msServiceProfiler（MindStudio Service Profiler）**](https://gitcode.com/Ascend/msserviceprofiler)<br>
    **服务化性能调优工具**：昇腾亲和的服务化性能调优工具，支持请求调度、模型执行过程可视化，提升服务化性能分析效率。

- [**msMemScope（MindStudio MemScope）**](https://gitcode.com/Ascend/msmemscope)<br>
    **内存工具**：针对昇腾显存调试调优场景的专用工具，提供整网级多维度显存数据采集、自动诊断、优化分析能力。

> [!WARNING] 注意
> 同时安装了CANN和msit，如果需要使用CANN中的组件，`source`CANN中的`set_env.sh`即可；如果需要使用msit中的组件，需要先`source`CANN中的`set_env.sh`，
> 再`source`msit才行。如果只`source`了msit的`set_env.sh`，上述组件会因为找不到CANN依赖而无法使用。

**PyPI 换源（可选）**

在安装msit的过程中可能会访问 PyPI 进行依赖包的下载和安装，您可以选择通过下列命令行进行 PyPI 换源。以下示例为华为云源，您可以替换为其他源：

```sh
pip3 config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple/
pip3 config set global.trusted-host repo.huaweicloud.com
```

## 安装步骤

### 安装软件包

1. 安装前需给run包添加可执行权限。

    ```shell
    chmod +x Ascend-mindstudio-inference-toolkit_linux-*.run
    ```

2. 执行以下命令安装。

    ```shell
    ./Ascend-mindstudio-inference-toolkit_linux-*.run --install
    ```

> [!NOTE] 说明
> 对于 root 用户，msit默认安装到 /usr/local/Ascend 目录下；如果使用普通用户进行安装，msit会默认安装到 ${HOME}/Ascend 下。<br>
> 如果要指定路径安装，则需添加 `--install-path`，例如下列代码会将msit安装到 `/path/to/install` 目录下：
>
> ```shell
> ./Ascend-mindstudio-inference-toolkit_linux-*.run --install --install-path=/path/to/install
> ```
>
> [!WARNING] 注意
> 如果要指定路径安装，则需添加 `--install-path`，但是需要注意：<br>
>
> 1. 使用 `--install-path=/path/to/install` 而非 `--install-path /path/to/install`，必须带上 `=`。
> 2. 指定的安装路径必须为绝对路径，不支持相对路径，输入相对路径会出现安装报错。

## 安装后配置

软件包安装成功后，工具会安装成功提示，为了确保工具正常运行，需设置环境变量。如下图展示的成功安装示例，设置环境变量的方法为 `source /opt/msit/set_env.sh`。

![msit install summary](../figures/msit_install_summary.png)

## 升级

如需使用run包替换运行环境中已安装的msit包，执行如下安装操作：

```sh
./Ascend-mindstudio-inference-toolkit_linux-*.run --upgrade
```

> [!NOTE] 说明
> 对于 root 用户，msit默认升级路径为 /usr/local/Ascend 目录；如果使用普通用户进行升级，msit会默认升级 ${HOME}/Ascend 下的工具包。<br>
> 如果要指定升级路径，则需添加 `--install-path`，使用方式和安装时一样。

## 卸载

软件包安装成功后，会在安装目录下生成 `msit_uninstall.sh` 文件，执行该文件即可进行卸载。如：

```sh
bash /usr/local/Ascend/msit/msit_uninstall.sh
```
