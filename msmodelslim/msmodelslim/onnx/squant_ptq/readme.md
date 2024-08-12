# 训练后量化（ONNX）
当前训练后量化工具自动对ONNX模型中的Conv和Gemm进行识别和量化，并将量化后的模型保存为.onnx文件，量化后的模型可以在推理服务器上运行，达到提升推理性能的目的。量化过程中用户需自行提供模型与数据集，调用API接口完成模型的量化调优。

ONNX模型量化包含Label-Free和Data-Free两种模式，均支持静态和动态shape模型的量化。Label-free模式下需要少量数据集矫正量化因子，Data-Free模式下无需数据集做矫正，可以直接对模型进行量化。

本readme指导用户调用Python API接口对模型进行Data-Free模式的识别和量化，并将量化后的模型保存为.onnx文件，量化后的模型可以在推理服务器上运行。

```python
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig

input_model = "model.onnx"  # 配置待量化模型的输入路径，请根据实际路径配置
output_path = "model_quant.onnx"  # 配置量化后模型的名称及输出路径，请根据实际路径配置

disable_names = []  # 需排除量化的节点名称，即手动回退的量化层名称。

config = QuantConfig(disable_names=disable_names,
                     quant_mode=0,
                     disable_first_layer=False,
                     disable_last_layer=False)

calib_data = []
calib = OnnxCalibrator(input_model, config, calib_data=calib_data)  # 使用OnnxCalibrator接口，输入待量化模型路径，量化配置数据
calib.run()  # 执行量化
calib.export_quant_onnx(output_path)  # 导出量化后模型
```