
# msModelSlim

## 介绍

MindStudio ModelSlim，昇腾模型压缩工具。 【Powered by MindStudio】


## 使用说明

msModelSlim当前处于逐步开源过程中，计划通过630(已发布),930(未发布),1230(未发布)三个版本进行过渡。

630版本只支持通过CANN使用，下载链接  
[arm 版本](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run?response-content-type=application/octet-stream)  
[x86 版本](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-toolkit_8.0.RC2_linux-x86_64.run?response-content-type=application/octet-stream) 

930版本支持通过CANN或开源方式使用，两边版本将保持一致，后续功能优化、新增将更新在开源版本中。  
版本交替期间提供两种方式使用msModelSlim工具：  
1、下载安装CANN并配置环境变量  
2、下载安装CANN并配置环境变量，下载安装开源版本msModelSlim，此场景会优先使用开源版本的工具
    操作步骤：
    1、通过git clone下载本仓代码
    2、设置CANN环境变量
    3、进入msmodelslim目录，运行脚本install.sh

1230版本只支持通过开源方式使用


#### 许可证
[Apache License 2.0](LICENSE)

