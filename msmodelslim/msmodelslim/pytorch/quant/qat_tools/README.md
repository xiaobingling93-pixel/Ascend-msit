1. 梯度补偿的训练量化算法介绍

本训练量化算法训练阶段可在昇腾910训练平台进行，在昇腾推理平台上部署，可将Float浮点模型转换为定点INT8模型，达到模型压缩、减少计算量、提升推理时延的目的。
本训练量化算法包含两种梯度补偿方案：（1）根据量化误差自适应的缩放量化值的梯度，从而实现更准确的梯度更新；（2）动态操作参数的梯度，确保一定数量的参数可以得到更新。
本算法可自动识别模型中的卷积和线性层（torch.nn.Linear和torch.nn.Conv2d）进行伪量化算子的插入，随后在训练平台进行微调，获得精度无损的伪量化模型。最终通过部署模块导出昇腾可部署的量化onnx。

1.1 伪量化训练及真量化部署
本算法的训练流程在NPU训练平台进行，在模型中识别可量化模块（conv/linear）插入伪量化节点，得到伪量化模型，随后进行微调。根据微调得到的量化参导出onnx模型部署至推理平台进行推理验证。

1.2 昇腾量化部署
利用Pytorch导出onnx的能力生成了浮点的ONNX模型，然后在浮点量化模型上找到量化节点，并修改ONNX插入量化和反向量节点。如浮点的ONNX节点，通过权重的name来寻找对应的量化参数，并在其前后插入支持NPU平台量化的AscendQuant和AscendDequant算子。

2. 使用指南

2.1 使用方法
2.1.1 环境变量配置
conda activate your_env 
source NPU相关环境变量

2.2.2 运行量化算法
1) 配置量化参数
QAT算法通过QatConfig来配置量化参数，可配置的量化参数如以下描述，均提供默认值，可支持典型业务模型的训练量化：
a) 量化算法配置：
w_bit：权重量化比特，默认为8，int类型。
is_forward：是否参考mmdetection对前向进行处理，默认为False，bool类型。
grad_scale：梯度补偿力度，默认为0， float类型。精度较差时建议0.001或者0.0005。
a_sym：激活值量化方法，默认为False，对应非对称量化。
amp_num：混合精度量化回退层数，要求格式int；默认为0，精度降低过多时，可增加回退层数，推荐优先回退1~2层，如果精度恢复不明显，再增加回退层数。
disable_names：手动回退的量化层名称，要求格式list[str]，如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等。
steps： 量化回退的步数，默认为1。
ignore_head_tail_node：是否忽略模型首尾节点不量化。默认False，表示不忽略，即首尾量化。
compressed_model_checkpoint：伪量化模型权重路径，默认为None，str类型。
opset_version：导出onnx模型时所选版本号，默认为11，int类型。
has_init_quant：是否做过量化初始化，默认为False，bool类型。
quant_mode：是否是量化模式，默认为True，bool类型。
save_params：导出时是否另外保存量化相关参数，默认为False，bool类型。
2) 执行量化算法
通过qsin_qat类封装量化算法插入伪量化算子，训练过程代码调用示例如下所示：
```
from msmodelslim.pytorch.quant.qat_tools import qsin_qat, QatConfig, get_logger

#insert fake quantization operator
quant_config = QatConfig()
quant_logger = get_logger()
model = qsin_qat(model, quant_config, quant_logger)

#finetune
train(model) #在训练过程中，需把权重ckpt文件保存，以备导出量化onnx使用
```
3) 导出量化onnx
训练阶段微调后保存伪量化模型权重，通过qsin_qat接口来导出昇腾可部署的量化onnx模型。涉及以下四个参数：
saved_ckpt：伪量化模型权重；
save_onnx_name：保存的onnx路径；
dummy_input：模型输入的shape，用于导出onnx模型时构造虚拟数据；
input_names： onnx的输入名称，有N个输入就要写N个名称，要求格式为list[str]
模型导出调用伪代码示例如下所示(以resnet为例)：
```
from msmodelslim.pytorch.quant.qat_tools import save_qsin_qat_model

save_onnx_name='dest.onnx'
dummy_input = torch.ones([batch_size, 3, 224, 224]).type(torch.float32)
saved_ckpt = 'saved_ckpt.pth'
input_names=['input']
save_qsin_qat_model(model, save_onnx_name, dummy_input, saved_ckpt, input_names)
```

注意事项：
1) 为保证精度，模型分类层和输入层不推荐量化，可在disable_names中配置分类层和输入层名称。
2) 如果量化精度不达标，可使用自动混合精度进行量化回退。