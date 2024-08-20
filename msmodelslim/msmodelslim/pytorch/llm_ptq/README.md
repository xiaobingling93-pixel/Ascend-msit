## LLM PTQ
LLM PTQ是一款专门为大模型设计的训练后量化工具，可以在无需训练成本的前提下，完成LLM大模型的训练后压缩并最大程度保障精度，一般需要在大模型下游任务评估流程打通的前提下，适配LLM PTQ量化工具代码进行模型量化和精度验证

## PTQ使用教程

本教程针对LLM大模型快速适配PTQ量化算法

教程分为4步：

1. 准备校准数据

2. 定义校准config

3. 执行PTQ量化校准 + 存储量化参数用于部署

4. 伪量化验证精度

 

### Step1.  准备校准数据
目前，我们需要根据用户处理数据脚本获取校准数据

原始模型预处理数据并推理：

    datasets = data_preprocessing(data_path) #数据预处理
    for inputs in datasets:
        output = model(**inputs) #推理
依据原始数据预处理过程，获取n条校准数据，用于后续校准，一般n取50即可：

      def get_dataset(num_calib=50):
          dataset_calib = []
          datasets = data_preprocessing(data_path)
          for _ in range(num_calib=50):
              data_calib.append(datasets[i])
          return dataset_calib

      + dataset_calib = get_dataset() #校准数据获取
### Step2.  定义校准config
      model = AutoModel.from_pretrained('THUDM/chatglm2-6b', torch_dtype=torch.float16, trust_remote_code=True).cpu() #模型导入
      + from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig #导入Calibrator和QuantConfig
      + quant_config = QuantConfig(dev_type='cpu', act_method=3, pr=0.5, mm_tensor=True, w_hessian=False) #定义校准config
### Step3.  执行PTQ量化校准 + 存储量化参数用于部署
      model = AutoModel.from_pretrained('THUDM/chatglm2-6b', torch_dtype=torch.float16, trust_remote_code=True).cpu()
      from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
      quant_config = QuantConfig(dev_type='cpu', act_method=3, pr=0.5, mm_tensor=True, w_hessian=False)
      + calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L1') 定义校准
      + calibrator.run() #执行PTQ量化校准
      + calibrator.save('save_path') #存储量化参数用于部署，在存储量化参数过程中，存在反序列化风险，已通过将保存的量化结果文件夹权限设置为750，将量化结果文件权限设置为400来消减该风险
### Step4.  伪量化验证精度
      model = AutoModel.from_pretrained('THUDM/chatglm2-6b', torch_dtype=torch.float16, trust_remote_code=True).cpu()
      from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
      quant_config = QuantConfig(dev_type='cpu', act_method=3, pr=0.5, mm_tensor=True, w_hessian=False)
      calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L1')
      calibrator.run()
      calibrator.save('save_path')
      + run_eval(model) #执行模型下游任务评估，验证伪量化精度
 

## ChatGLM_V1-6B、ChatGLM_V2-6B、LLaMA-13B量化配置 + 使用简介

### ChatGLM_V1-6B 
量化config:

    QuantConfig(disable_names=[], dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False)

使用方式：

    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()
    calibrator.save(qaunt_weight_save_path)

### ChatGLM_V2-6B 
量化config:

    QuantConfig(disable_names=[], dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False)

使用方式：

    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L2')
    calibrator.run()
    calibrator.save(qaunt_weight_save_path)

### LLaMA-13B 
量化config:

    QuantConfig(disable_names=[], dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False)

使用方式：

    anti_config = AntiOutlierConfig(anti_method="m2", dev_type='cpu')
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()
    calibrator.save(qaunt_weight_save_path)

## 工具参数简介
### 一、LLM量化介绍
#### 量化config:
+ disable_names: 为保障精度，不做量化的层的名称列表，为保障性能，尽量设置为空列表[]
+ dev_type: 设备类型，默认设置为'cpu'，可选用'npu'
+ dev_id: 设备号，输入为int值。如果在npu上进行计算，则输入模型所在设备号`model.device.index`
+ act_method: 激活值量化方法,共有1,2,3三种，分别对应min-max量化方式、histogram量化方式、自动混合量化方式，LLM大模型场景建议使用方式3
+ pr:量化正则百分比,目前的模型最优值0.5(根据内部实验，0.5为最优值)，增大减小均会降低量化效果，建议试使用0.5
+ mm_tensor:权重量化方式。True: per-tensor量化； False: per-channel量化，在大模型场景，使用per-channel
#### Calibrator类参数传入：
+ model：用于量化的模型
+ quant_config：量化config，使用llm_ptq_tools中QuantConfig构建
+ calib_data：用于LLM大模型量化校准的数据，在npu上计算时需要注意数据与model所在设备保持一致
+ disable_level：自动回退等级，默认为'L0'，在模型精度损失大可以适当提升等级，例如'L5'
### 二、离群值抑制anti_outlier介绍
#### 离群值抑制config
+ anti_method: 异常值抑制anti_outlier使用方法，支持'm1'、'm2'两种模式(内部调优中，建议使用'm2')，可参考smoothquant
+ dev_type: 设备类型，设置为'cpu'
#### AntiOutlier类参数传入：
+ model: 用于大模型离群值抑制的模型
+ calib_data: 用于离群值抑制的校准数据
+ cfg: 离群值抑制config，使用anti_outlier中的AntiOutlierConfig构建
+ norm_class_name: 数据类型：str。默认为None，由于每个模型结构可能存在差异，系统自动识别norm无法涵盖所有结构,若系统自动识别norm失败，则需要用户手动输入自定义的norm类名，例如norm_class_name = 'LlamaRMSNorm'。
