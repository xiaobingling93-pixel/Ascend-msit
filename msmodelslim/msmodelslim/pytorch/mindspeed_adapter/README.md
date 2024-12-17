## mindspeed适配器
原有的llm_ptq模块主要支持基于transformers框架下的大模型量化压缩功能，本模块提供了针对modellink模型的量化适配器，可以直接量化mindspeed-LLM模型

### 前提条件
- 仅支持在以下产品中使用。
    - Atlas 训练系列产品。
    - Atlas A2训练系列产品/Atlas 800I A2推理产品。

- 已参考环境准备，完成CANN开发环境的部署、以及PyTorch 2.1.0及以上版本的框架和npu插件、Python环境变量配置。
- 大模型量化工具须执行命令安装如下依赖。
  如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。
```
pip3 install numpy==1.25.2
pip3 install transformers        #需大于等于4.29.1版本，LLaMA模型需指定安装4.29.1版本
pip3 install accelerate==0.21.0  #若需要使用NPU多卡并行方式对模型进行量化，需大于等于0.28.0版本
pip3 install tqdm==4.66.1
```
安装mindspeed-LLM库

### 功能约束
当前模型适配器仅验证过支持w8a8的量化，以及异常值抑制模块的m3和m5算法，仅支持NPU执行量化，不支持CPU量化