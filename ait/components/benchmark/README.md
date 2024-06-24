# ais_bench推理工具使用指南

ais_bench推理工具，用来针对指定的推理模型运行推理程序，并能够测试推理模型的性能（包括吞吐率、时延）。

该部分代码已移至 [Gitee Ascend/tools/ais-bench_workload](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)，文档及安装方法可参照对应说明

#### 下载whl包安装

1. 下载如下aclruntime和ais_bench推理程序的whl包。

   0.0.2版本（aclruntime包请根据当前环境选择适配版本）：

   |whl包|commit节点|MD5|SHA256|
   |---|---|---|---|
   |[aclruntime-0.0.2-cp37-cp37m-linux_x86_64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp37-cp37m-linux_x86_64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|E14ACDFBDD52E08F79456D9BC72D589C| F1523E25B714EF51E03D640570E8655A139DB8B9340C8DD6E4DA82D6122B2C01|
   |[aclruntime-0.0.2-cp37-cp37m-linux_aarch64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp37-cp37m-linux_aarch64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e| 9455E267118011CAC764ECECA3B13B64|4C1F7CD1CD767912B597EAF4F4BE296E914D43DE4AF80C6894399B7BF313A80F|
   |[aclruntime-0.0.2-cp38-cp38-linux_x86_64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp38-cp38-linux_x86_64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|CE23FEDB8BAC2917E7238B8E25F8E54D| 63C86CEE2C9F622FAB2F6A1AA4EAB47D2D68622EC12BDC8F74A9F8CED6506D67|
   |[aclruntime-0.0.2-cp38-cp38-linux_aarch64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp38-cp38-linux_aarch64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|52CA43514A7373E50678A890D085C531|20AFB7A24DB774EF67250E062A0F593E419DBC5A1A668B98B60D4BBF8CA87E88|
   |[aclruntime-0.0.2-cp39-cp39-linux_x86_64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp39-cp39-linux_x86_64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|55016F7E2544849E128AA7B5A608893D| 22824F38CAA547805FA76DBAA4889307BE171B79CCDA68AD00FED946762E6EAD|
   |[aclruntime-0.0.2-cp39-cp39-linux_aarch64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp39-cp39-linux_aarch64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|080065E702277C1EE443B02C902B49E6|258CDCFBBA145E200D08F1976C442BC921D68961157BDFD1F0D73985FDC45F24|
   |[aclruntime-0.0.2-cp310-cp310-linux_x86_64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp310-cp310-linux_x86_64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|78242C34E7DB95E6587C47254E309BBB|4F563603FCFF9CBC3FF74322936894C0E01038BF0101E85F03975B8BDDC57E6A|
   |[aclruntime-0.0.2-cp310-cp310-linux_aarch64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp310-cp310-linux_aarch64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|5988B1565C8136BF17374FA703BE0BC7|185CBC5DDE9C03E26494871FCC0A6F91351DE654CB36F9438DDBF9637C049CB8|
   |[ais_bench-0.0.2-py3-none-any.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|1E43A8BE245C015B47C9C5E72EA5F619|D52406D0AC02F9A8EBEFDCE0866736322753827298A4FCB1C23DA98789BF8EFE|


2. 执行如下命令，进行安装。

   ```bash
   # 安装aclruntime
   pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
   # 安装ais_bench推理程序
   pip3 install ./ais_bench-{version}-py3-none-any.whl
   ```

   {version}表示软件版本号，{python_version}表示Python版本号，{arch}表示CPU架构。

   说明：若为覆盖安装，请增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl --force-reinstall
   pip3 install ./ais_bench-{version}-py3-none-any.whl --force-reinstall
   ```

   分别提示如下信息则表示安装成功：

   ```bash
   # 成功安装aclruntime
   Successfully installed aclruntime-{version}
   # 成功安装ais_bench推理程序
   Successfully installed ais_bench-{version}
   ```


#### 一键式编译安装

1. **安装aclruntime包**

   在安装环境执行如下命令安装aclruntime包：

   ```bash
   pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend'
   ```

   说明：若为覆盖安装，请增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install -v --force-reinstall 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend'
   ```

   提示如下示例信息则表示安装成功：

   ```bash
   Successfully installed aclruntime-{version}
   ```

2. **安装ais_bench推理程序包**

   在安装环境执行如下命令安装ais_bench推理程序包：

   ```bash
   pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=ais_bench&subdirectory=ais-bench_workload/tool/ais_bench'
   ```

   说明：若为覆盖安装，请增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install -v --force-reinstall 'git+https://gitee.com/ascend/tools.git#egg=ais_bench&subdirectory=ais-bench_workload/tool/ais_bench'
   ```

   提示如下示例信息则表示安装成功：

   ```bash
   Successfully installed ais_bench-{version}
   ```