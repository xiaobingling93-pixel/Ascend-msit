# msit安全声明

## 系统安全加固

建议用户在系统中配置开启ASLR（级别2 ），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：

    echo 2 > /proc/sys/kernel/randomize_va_space

## 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用msit、msmodelslim、msserviceprofiler、msprechecker工具。

## 文件权限控制

1. 用户向工具提供输入文件输入时，建议提供的文件属主与工具进程属主一致，且文件权限他人不可修改（包括group、others）。工具落盘文件权限默认他人不可写，用户可根据需要自行对生成后的相关文件进行权限控制。

2. 用户安装和使用过程需要做好权限控制，建议参考文件权限参考进行设置。


##### 文件权限参考

| 类型                               | Linux权限参考最大值 |
| ---------------------------------- | ------------------- |
| 用户主目录                         | 750（rwxr-x---）    |
| 程序文件(含脚本文件、库文件等)     | 550（r-xr-x---）    |
| 程序文件目录                       | 550（r-xr-x---）    |
| 配置文件                           | 640（rw-r-----）    |
| 配置文件目录                       | 750（rwxr-x---）    |
| 日志文件(记录完毕或者已经归档)     | 440（r--r-----）    |
| 日志文件(正在记录)                 | 640（rw-r-----）    |
| 日志文件目录                       | 750（rwxr-x---）    |
| Debug文件                          | 640（rw-r-----）    |
| Debug文件目录                      | 750（rwxr-x---）    |
| 临时文件目录                       | 750（rwxr-x---）    |
| 维护升级文件目录                   | 770（rwxrwx---）    |
| 业务数据文件                       | 640（rw-r-----）    |
| 业务数据文件目录                   | 750（rwxr-x---）    |
| 密钥组件、私钥、证书、密文文件目录 | 700（rwx------）    |
| 密钥组件、私钥、证书、加密密文     | 600（rw-------）    |
| 加解密接口、加解密脚本             | 500（r-x------）    |

## 数据安全声明

​    1、工具使用过程中需要加载和保存数据，部分接口直接或间接使用风险模块pickle，可能存在数据风险，如torch.load等接口，可参考[torch.load](https://pytorch.org/docs/main/generated/torch.load.html#torch.load)了解具体风险。

​    2、onnx模型加载解析特色功能依赖第三方onnx，小于1.15.0版本容易受到越界读取的攻击，使用前要先确保加载的onnx模型可信。

## 构建安全声明

​    msit、msmodelslim支持源码编译安装，在编译时会下载依赖第三方库并执行构建shell脚本，在编译过程中会产生临时程序文件和编译目录。用户可根据需要自行对源代码目录内的文件进行权限管控降低安全风险，用户在构建过程中可根据需要修改构建脚本以避免相关安全风险，并注意构建结果的安全。

## 运行安全声明

1. 工具加载数据集时，如数据集加载内存大小超出内存容量限制，可能引发错误并导致进程意外退出；采集时间过长导致生成数据超过磁盘空间大小时，可能会导致异常退出。
2. 工具在运行异常时会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括查看日志文件，采集解析过程中生成的结果文件等方式。

## 公网地址声明

在msit仓工具的配置文件和脚本中存在的[公网地址](#公网地址)

##### 公网地址

| 类型     | 开源代码地址 | 文件名                                        | 公网IP地址/公网URL地址/域名/邮箱地址                   | 用途说明                   |
| -------- | ------------ | --------------------------------------------- | ------------------------------------------------------ | -------------------------- |
| 开源软件 | -            | 所有文件                                       | http://www.apache.org/licenses/LICENSE-2.0             | 文件头中的license信息说明  |
| 开源软件 | -            | msprechecker/pyproject.toml                   | https://gitee.com/ascend/msit/tree/master/msprechecker | msprechecker工具的仓库地址 |
| 开源软件 | -            | msprechecker/pyproject.toml                   | https://gitee.com/ascend/msit/issues                   | msit仓的issue地址          |
| 开源软件 | -            | msmodelslim/config/config.ini                 | https://gitee.com/ascend/msit                          | msit仓的地址               |
| 开源软件 | -            | msmodelslim/example/osp1_2/model/scheduler.py | https://arxiv.org/abs/2305.08891                       | 注释中的公网地址           |
| 开源软件 | - | msit/install.sh | http://mirrors.huaweicloud.com/repository/pypi/simple | 若pip源为华为云，则优先安装skl2onnx(当前mirrors.huaweicloud.com中skl2onnx已停止更新  不包含1.14.1及以上版本) |
| 开源软件 | - | msit/install.sh | https://mirrors.huaweicloud.com/repository/pypi/simple | 若pip源为华为云，则优先安装skl2onnx(当前mirrors.huaweicloud.com中skl2onnx已停止更新，不包含1.14.1及以上版本) |
| 开源软件 | - | msit/install.sh | https://mirrors.tools.huawei.com/pypi/simple | 若pip源为华为云，则优先安装skl2onnx(当前mirrors.huaweicloud.com中skl2onnx已停止更新，不包含1.14.1及以上版本) |
| 开源软件 | - | msit/setup.py | https://gitee.com/ascend/msit | 用于msit工具打包成python包的主页说明 |
| 开源软件 | - | msit/components/__main__.py | https://gitee.com/ascend/msit/wikis/Home | msit工具wiki主页说明 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | http://.*archive.ubuntu.com | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | http://repo.huaweicloud.com | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | http://ports.ubuntu.com | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | https://repo.huaweicloud.com/repository/pypi/simple | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | https://gitee.com/ascend/msit.git | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | https://pypi.tuna.tsinghua.edu.cn/simple/ | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/debug/compare/setup.py | https://gitee.com/ascend/msit/tree/master/msit/components/debug/compare | 用于compare组件打包成python包的主页说明 |
| 开源软件 | - | msit/components/debug/compare/tests/get_pth_resnet18_model.sh | https://download.pytorch.org/models/resnet18-f37072fd.pth | 用于下载compare组件测试用例所需的开源模型 |
| 开源软件 | - | msit/components/debug/compare/tests/get_pth_resnet18_model.sh | https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet18_for_PyTorch/resnet18_pth2onnx.py | 用于下载compare组件测试用例所需的开源模型 |
| 开源软件 | - | msit/components/llm/README.md | https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/msit-0.0.1-py3-none-linux_aarch64.whl | msit预编译whl包下载地址 |
| 开源软件 | - | msit/components/llm/README.md | https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/msit-0.0.1-py3-none-linux_x86_64.whl | msit预编译whl包下载地址 |
| 开源软件 | - | msit/components/llm/README.md | https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_aarch64.whl | msit_llm预编译whl包下载地址 |
| 开源软件 | - | msit/components/llm/README.md | https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI1/ait_llm-0.1.0-py3-none-linux_aarch64.whl | msit_llm预编译whl包下载地址 |
| 开源软件 | - | msit/components/llm/README.md | https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_x86_64.whl | msit_llm预编译whl包下载地址 |
| 开源软件 | - | msit/components/llm/README.md | https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI1/ait_llm-0.1.0-py3-none-linux_x86_64.whl | msit_llm预编译whl包下载地址 |
| 开源软件 | - | msit/components/llm/README.md | https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240117/ait_llm-0.2.0-py3-none-linux_aarch64.whl | msit_llm预编译whl包下载地址 |
| 开源软件 | - | msit/components/llm/README.md | https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240117/ait_llm-0.2.0-py3-none-linux_x86_64.whl | msit_llm预编译whl包下载地址 |
| 开源软件 | - | msit/examples/cli/debug/compare/12_pta_acl_cmp_weight_map/matched_pie.png | https://matplotlib.org/ | 样例图片文件中的来源 |
| 开源软件 | - | msit/components/benchmark/test/test.sh | https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame | aisbench obs桶上存放的下x86_64架构上使用的msame推理工具，用于在DT代码中和benchmark推理工具的推理效果进行比较 |
| 开源软件 | - | msit/components/benchmark/test/test.sh | https://aisbench.obs.myhuaweicloud.com/packet/msame/arm/msame | aisbench obs桶上存放的下aarch64架构上使用的msame推理工具，用于在DT代码中和benchmark推理工具的推理效果进行比较 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_crnn_data.sh | https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/c-version/CRNN_for_PyTorch/zh/1.3/m/CRNN_for_PyTorch_1.3_model.zip | 用于DT脚本，ascend-repo-modelzoo obs桶中存放的crnn的onnx模型文件（该文件在压缩包中） |
| 开源软件 | - | msit/components/benchmark/test/get_pth_crnn_data.sh | https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/aipp_resnet50.aippconfig | 用于DT脚本，Ascend/modelzoo gitee repo中存放的resnet50模型在atc转换时需要用到的aipp配置文件 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_crnn_data.sh | https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet101_Pytorch_Infer/resnet101_pth2onnx.py" | 用于DT脚本，Ascend/modelzoo gitee repo中存放的resnet101模型从pth格式转换成onnx格式的转换脚本 |
| 开源软件 | - | msit/components/benchmark/test/get_bert_data.sh | https://ascend-repo-modelzoo.obs.myhuaweicloud.com/model/ATC%20BERT_BASE_SQuAD1.1%28FP16%29%20from%20Tensorflow-Ascend310/zh/1.1/ATC%20BERT_BASE_SQuAD1.1%28FP16%29%20from%20Tensorflow-Ascend310.zip | 用于DT脚本，ascend-repo-modelzoo obs桶中存放的bert的pb模型文件（该文件在压缩包中） |
| 开源软件 | - | msit/components/benchmark/test/get_pth_resnet101_data.sh | https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet101_Pytorch_Infer/resnet101_pth2onnx.py | 用于DT脚本，Ascend/modelzoo gitee repo中存放的resnet101模型从pth格式转换成onnx格式的转换脚本 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_resnet101_data.sh | https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/aipp_resnet50.aippconfig | 用于DT脚本，Ascend/modelzoo gitee repo中存放的resnet50模型在atc转换时需要用到的aipp配置文件 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_resnet101_data.sh | https://download.pytorch.org/models/resnet101-63fe2227.pth | 用于DT脚本，pytorch官网存放的resnet101的pth模型文件 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_resnet50_data.sh | https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/pth2onnx.py | 用于DT脚本，Ascend/modelzoo gitee repo中存放的resnet50模型从pth格式转换成onnx格式的转换脚本 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_resnet50_data.sh | https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/aipp_resnet50.aippconfig | 用于DT脚本，Ascend/modelzoo gitee repo中存放的resnet50模型在atc转换时需要用到的aipp配置文件 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_resnet50_data.sh | https://download.pytorch.org/models/resnet50-0676ba61.pth | 用于DT脚本，pytorch官网存放的resnet50的pth模型文件 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_inception_v3_data.sh | https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/InceptionV3_for_Pytorch/aipp_inceptionv3_pth.config | 用于DT脚本，Ascend/modelzoo gitee repo中存放的inceptionv3模型在atc转换时需要用到的aipp配置文件 |
| 开源软件 | - | msit/components/benchmark/test/get_pth_inception_v3_data.sh | https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/InceptionV3/inceptionv3.onnx | 用于DT脚本，obs-9be7 obs桶中存放的inceptionv3的onnx模型文件 |
| 开源软件 | - | msit/components/benchmark/test/get_yolo_data.sh | https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/model/ATC%20Yolov3%20from%20Pytorch%20Ascend310/zh/1.1/ATC_Yolov3_from_Pytorch_Ascend310.zip | 用于DT脚本，obs-9be7 obs桶中存放的yolov3的onnx模型文件（该文件在压缩包中） |
| 开源软件 | - | msit/components/benchmark/ais_bench/infer/miscellaneous.py | https://gitee.com/ascend/msit/tree/master/msit/components/benchmark | 用于指引用户安装benchmark缺失的aclruntime包，Ascend/msit gitee repo中benchmark组件的主页文档 |
| 开源软件 | - | msit/components/benchmark/ais_bench/infer/path_security_check.py | https://gitee.com/ascend/msit/wikis/msit_security_error_log_solution | 用于指引用户使用benchmark工具输入非法信息出现报错的解决方法，Ascend/msit repo的wiki |
| 开源软件 | - | msit/components/benchmark/ais_bench/evaluate/dataset/download.sh | https://llm-dataset.obs.myhuaweicloud.com/$dataset.tar.gz | 用于获取数据集，llm-dataset obs桶中存放的数据集的压缩包 |
| 开源软件 | - | msit/components/benchmark/ais_bench/backend/setup.py | https://github.com/pybind/python_example/pull/53 | 注释中说明pylind11使用的参考对象，github pybind repo中pybind11的python使用样例 |
| 开源软件 | - | msit/components/llm/setup.py | https://gitee.com/ascend/msit/msit/components/llm | 用于msit-llm工具打包成python包的主页说明 |
| 开源软件 | - | msit/components/debug/surgeon/setup.py | https://gitee.com/ascend/msit.git | msit的gitee仓地址 |
| 开源软件 | - | msit/components/debug/surgeon/docs/img/inference.png | www.iodraw.com | 样例图片文件中的来源 |
| 开源软件 | - | msit/components/debug/surgeon/test/dataset/test_img/cat.jpg | http://ns.adobe.com/xap/1.0/ | 样例图片文件中的来源 |
| 开源软件 | - | msit/components/convert/setup.py | https://gitee.com/ascend/msit | msit的gitee仓地址 |
| 开源软件 | - | msit/components/analyze/setup.py | https://gitee.com/ascend/msit | msit的gitee仓地址 |
| 开源软件 | - | msit/components/transplt/config.ini | https://msit-resources.obs.cn-south-1.myhuaweicloud.com/config.zip | msit transplt的配置文件下载地址 |
| 开源软件 | - | msit/components/transplt/config.ini | https://msit-resources.obs.cn-south-1.myhuaweicloud.com/float.h | mingw的windows安装包patch文件下载地址 |
| 开源软件 | - | msit/components/transplt/config.ini | https://msit-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip | msit transplt的配置文件下载地址 |
| 开源软件 | - | msit/components/transplt/config.ini | https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.0/LLVM-12.0.0-win64.exe | LLVM的官网下载地址 |
| 开源软件 | - | msit/components/transplt/config.ini | https://nchc.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/8.1.0/threads-posix/seh/x86_64-8.1.0-release-posix-seh-rt_v6-rev0.7z | mingw官网下载地址 |
| 开源软件 | - | msit/components/transplt/config.ini | https://www.7-zip.org/a/7z2301-x64.msi | 7-zip官网下载地址 |
| 开源软件 | - | msit/components/transplt/Dockerfile | http://apt.llvm.org/llvm-snapshot.gpg.key | LLVM的官方gpg key下载地址 |
| 开源软件 | - | msit/components/transplt/Dockerfile | https://msit-resources.obs.cn-south-1.myhuaweicloud.com/config.zip | msit transplt的配置文件下载地址 |
| 开源软件 | - | msit/components/transplt/Dockerfile | https://msit-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip | msit transplt的配置文件下载地址 |
| 开源软件 | - | msit/components/transplt/Dockerfile | https://bootstrap.pypa.io/get-pip.py | pip安装脚本官网下载地址 |
| 开源软件 | - | msit/components/transplt/Dockerfile | https://gitee.com/ascend/msit.git | msit的gitee仓地址 |
| 开源软件 | - | msit/components/transplt/Dockerfile | https://mirrors.cernet.edu.cn/llvm-apt/focal/ | ubuntu的llvm源地址 |
| 开源软件 | - | msit/components/transplt/Dockerfile | https://repo.huaweicloud.com/repository/pypi/simple | 华为云官方pip源地址 |
| 开源软件 | - | msit/components/transplt/install.sh | http://apt.llvm.org/llvm-snapshot.gpg.key | LLVM的官方gpg key下载地址 |
| 开源软件 | - | msit/components/transplt/install.sh | http://mirrors.163.com/openSUSE/distribution/leap/15.1/repo/non-oss | openSUSE软件包源地址 |
| 开源软件 | - | msit/components/transplt/install.sh | http://mirrors.163.com/openSUSE/distribution/leap/15.1/repo/oss | openSUSE软件包源地址 |
| 开源软件 | - | msit/components/transplt/install.sh | http://mirrors.163.com/openSUSE/update/leap/15.1/non-oss | openSUSE软件包源地址 |
| 开源软件 | - | msit/components/transplt/install.sh | http://mirrors.163.com/openSUSE/update/leap/15.1/oss | openSUSE软件包源地址 |
| 开源软件 | - | msit/components/transplt/install.sh | https://msit-resources.obs.cn-south-1.myhuaweicloud.com/config.zip | msit transplt的配置文件下载地址 |
| 开源软件 | - | msit/components/transplt/install.sh | https://msit-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip | msit transplt的配置文件下载地址 |
| 开源软件 | - | msit/components/transplt/install.sh | https://mirrors.cernet.edu.cn/llvm-apt/focal/ | ubuntu的llvm源地址 |
| 开源软件 | - | msit/components/transplt/setup.py | https://gitee.com/ascend/msit | msit的gitee仓地址 |
| 开源软件 | - | msit/components/transplt/test/test.sh | https://msit-resources.obs.cn-south-1.myhuaweicloud.com | msit transplt的配置文件下载地址 |
| 开源软件 | - | msit/components/transplt/tools/update_link/config.json | "https://www.hiascend.com/document/detail/zh/canncommercial//inferapplicationdev/aclcppdevg/aclcppdevg_03_%04d.html | " | 昇腾社区cann API地址 |
| 开源软件 | - | msit/components/transplt/tools/update_link/config.json | "https://www.hiascend.com/document/detail/zh/mind-sdk//vision/mxvisionug/mxvisionug_%04d.html | " | 昇腾社区mind sdk API地址 |
| 开源软件 | - | msit/components/transplt/tools/update_link/src/web_crawler.py | https://www.hiascend.com | 昇腾社区地址 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | http://.*security.ubuntu.com | 构建compare组件的docker安装镜像  |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | https://mirrors.huaweicloud.com | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/debug/compare/Dockerfile | repo.huaweicloud.com | 构建compare组件的docker安装镜像 |
| 开源软件 | - | msit/components/config/config.ini | https://gitee.com/ascend/msit | msit的gitee仓地址 |
| 开源软件 | - | msit/components/config/config.ini | https://gitee.com/ascend/msit/msit/components/llm | msit llm工具仓库地址 |
| 开源软件 | - | msit/components/config/config.ini | https://gitee.com/ascend/msit/tree/master/msit/components/debug/compare | msit debug compare工具仓库地址 |
| 开源软件 | - | msit/components/config/config.ini | https://gitee.com/ascend/msit/tree/master/msit/components/debug/opcheck | msit debug opcheck工具仓库地址 |
| 开源软件 | - | msit/components/config/config.ini | https://github.com/nlohmann/json/archive/refs/tags | 第三方库nlohmann地址 |
| 开源软件 | - | msit/components/config/config.ini | https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ | aisbench工具包地址 |
| 开源软件 | - | msit/components/config/config.ini | https://gitee.com/ascend/tools.git | aisbench工具所在的tools仓地址 |
| 开源软件 | - | msit/components/config/config.ini | http://mirrors.huaweicloud.com/repository/pypi/simple | PyPI华为云镜像地址 |
| 开源软件 | - | msit/components/config/config.ini | https://mirrors.huaweicloud.com/repository/pypi/simple | PyPI华为云镜像地址 |
| 开源软件 | - | msit/components/config/config.ini | https://mirrors.tools.huawei.com/pypi/simple | PyPI镜像地址 | 
| 开源软件 | - | msit/components/config/config.ini | http://www.apache.org/licenses/LICENSE-2.0 | license信息说明 |


## 公开接口声明（待完善）

msit是xxx，支持xxx。支持用户使用xx接口，具体可参考xxx。

msit提供了对外的自定义接口。如果一个函数看起来符合公开接口的标准且在文档中有展示，则该接口是公开接口。否则，使用该功能前可以在社区询问该功能是否确实是公开的或意外暴露的接口，因为这些未暴露接口将来可能会被修改或者删除。

msit项目采用C++和Python联合开发，当前除xxx场景外正式接口只提供Python接口，动态库中的接口不直接提供服务，暴露的接口为内部使用，不建议用户使用。

## 通信安全加固

暂不涉及远程通信，建议用户在有防火墙或本地局域网的安全网络环境中使用工具，并注意其他三方软件的通信安全。

## 通信矩阵

简述下情况

### 通信矩阵信息

| 序号 | 代码仓            | 功能           | 源设备                       | 源IP                           | 源端口                                                       | 目的设备   | 目的IP               | 目的端口 （侦听） | 协议 | 端口说明                                                     | 端口配置 | 侦听端口是否可更改 | 认证方式 | 加密方式 | 所属平面 | 版本     | 特殊场景 | 备注 |
| ---- | ----------------- | -------------- | ---------------------------- | ------------------------------ | ------------------------------------------------------------ | ---------- | -------------------- | ----------------- | ---- | ------------------------------------------------------------ | -------- | ------------------ | -------- | -------- | -------- | -------- | -------- | ---- |
| 1    | msserviceprofiler | vllm服务端通信 | 自动寻优工具拉起的VLLM服务端 | 推理服务port参数对应的IP地址。 | 根据现网客户实际要求配置的固定端口，对应起服务时的--port字段，默认配置为8000。 | vllm客户端 | vllm客户端通信ip地址 | 默认8000          | HTTP | 工具将通过命令行拉起客户环境中的推理服务，若客户未进行配置，则默认端口8000，否则使用客户配置指定端口拉起vllm服务 | 不涉及   | 可修改             | 不涉及   | 不涉及   | 数据面   | 所有版本 | 无       |      |