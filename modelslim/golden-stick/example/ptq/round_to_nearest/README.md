## LLM PTQ
LLM PTQ是一款专门为大模型设计的训练后量化工具，可以在无需训练成本的前提下，完成LLM大模型的训练后压缩并最大程度保障精度，一般需要在大模型下游任务评估流程打通的前提下，适配LLM PTQ量化工具代码进行模型量化和精度验证

## PTQ使用教程

本教程针对LLM大模型快速适配PTQ量化算法

教程分为4步：

1. 定义校准config

2. 执行PTQ量化校准 + 存储量化参数


### Step1.  定义校准config
      network = LlamaForCausalLM(quant_config.cfg.model.model_config) #模型导入
      + from Mindspore.quant.ptq_quant.quant_config import QuantConfig #导入QuantConfig
      + from Mindspore.quant.ptq_quant.quant_tools import Calibrator #导入Calibrator
      + quant_config = QuantConfig(uargs.config_path, uargs.network, uargs.framework, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND) #定义校准config
### Step2.  执行PTQ量化校准 + 存储量化参数用于部署
      network = LlamaForCausalLM(quant_config.cfg.model.model_config) #模型导入
      from Mindspore.quant.ptq_quant.quant_config import QuantConfig #导入QuantConfig
      from Mindspore.quant.ptq_quant.quant_tools import Calibrator #导入Calibrator
      quant_config = QuantConfig(uargs.config_path, uargs.network, uargs.framework, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND) #定义校准config
      + Calibrator = Calibrator(network, quant_config.PTQcfg) 定义校准
      + calibrator.run() #执行PTQ量化校准
      + calibrator.save('save_path') #存储量化参数用于部署


## 启动脚本
--config_path
    config文件路径
--fp_ckpt_path
    浮点模型ckpt文件路径
--save_ckpt_path
    量化模型ckpt权重文件保存路径
--framework
    框架格式，支持pytorch:"torch"和mindspore:"ms"
--network
    网络结构，支持"llama"
示例
    python llama2_w8a16_quant.py -c "./run_llama2_7b_910b.yaml"  -f "ms" -k "/your/path/llama2_7b.ckpt" -s "./llama2_quant_save_path" -n llama2_7b