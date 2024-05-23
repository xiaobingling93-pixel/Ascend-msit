
# 加速库模型数据dump

提供大模型推理过程中产生的中间数据的dump能力，如tensor、model拓扑信息、layer拓扑信息、算子信息、cpu_profiling数据等。

## 使用方式

```bash
ait llm dump --exec "bash run.sh patches/models/modeling_xxx.py" [可选参数]
```
## 命令行参数

| 参数名                         | 描述                                                         | 必选 |
| ------------------------------ | ------------------------------------------------------------ | ---- |
| --exec                         | 指定拉起执行大模型推理脚本的命令，使用示例： --exec "bash run.sh patches/models/modeling_xxx.py"。**注：命令中不支持重定向字符，如果需要重定向输出，建议将执行命令写入shell脚本，然后启动shell脚本。** | 是   |
| --type                         | dump类型，可选范围：['model', 'layer', 'op', 'kernel', 'tensor', 'cpu_profiling', 'onnx']，分别表示保存model拓扑信息、layer拓扑信息、算子信息、kernel算子信息、tesnor数据、cpu_profiling数据、onnx模型。其中'onnx'需要和'model'、'layer'组合使用，用于将model和layer的拓扑信息转换成onnx，可视化模型结构。默认为['tensor']。使用方式：--type layer tensor | 否   |
| -sd，--only-save-desc          | 只保存tensor描述信息开关，默认为否，开启开关时将dump tensor的描述信息，使用方式：-sd | 否   |
| -ids，--save-operation-ids     | 设置dump指定id的算子的tensor，默认为空，全量dump。使用方式：-ids 2, 3_1 表示只dump第2个operation和第3个operation的第1个算子的数据，id从0开始。若不确定算子id，可以先执行ait llm dump --exec xx --type model命令，将model信息dump下来，即可获得模型中所有的算子id信息。 | 否   |
| -er，--execute-range           | 指定dump的token轮次范围，区间左右全闭，可以支持多个区间序列，默认为第0次，使用方式：-er 1,3 或 -er 3,5,7,7（代表区间[3,5],[7,7],也就是第3，4，5，7次token） | 否   |
| -child，--save-operation-child | 选择是否dump所有子操作的tensor数据，仅使用ids场景下有效，默认为true。使用方式：-child True | 否   |
| -time，--save-time             | 选择保存的时间节点，取值[0,1,2]，0代表保存执行前(before)，1代表保存执行后(after)，2代表前后都保存。默认保存after。使用方式：-time 0 | 否   |
| -opname，--operation-name      | 指定需要dump的算子类型，只需要指定算子名称的开头，可以模糊匹配，如selfattention只需要填写self。使用方式：-opname self | 否   |
| -tiling，--save-tiling         | 选择是否需要保存tiling数据，默认为false。使用方式：-tiling   | 否   |
| --save-tensor-part, -stp       | 指定保存tensor的部分，0为仅intensor，1为仅outtensor，2为全部保存，默认为2。使用示例：-stp 1 | 否   |
| -o, --output                   | 指定dump数据的输出目录，默认为'./'，使用示例：-o aasx/sss    | 否   |
| -device, --device-id           | 指定dump数据的device id，默认为 None 表示不限制。如指定 --device-id 1，将只dump 1卡的数据 | 否   |
| -l, --log-level                | 指定log level，默认为 info，可选值 debug, info, warning, error, fatal, critical | 否   |

## 结果查看
### Dump 落盘位置

Dump默认落盘路径 `{DUMP_DIR}`在当前目录下，如果指定output目录，落盘路径则为指定的 `{OUTPUT_DIR}`。

注：`{device_id}`为设备号；`{PID}`为进程号；`{TID}`为 `token_id`；`{TIMESTAMP}`为时间戳；`{executeCount}`为 `operation`运行次数。

- tensor信息，具体路径是 `{DUMP_DIR}/ait_dump/tensors/{device_id}_{PID}/{TID}`目录下(使用老版本的cann包可能导致tensor落盘路径不同）。
- layer信息，具体路径是 `{DUMP_DIR}/ait_dump/layer/{PID}`目录下。
- model信息，具体路径是 `{DUMP_DIR}/ait_dump/model/{PID}`目录下。注：由于model由layer组合而成，因此使用model时，默认同时会落盘layer信息。
- onnx需要和layer、model配合使用，落盘位置和model、layer相同的目录。
- cpu_profiling信息，具体路径是 `{DUMP_DIR}/ait_dump/cpu_profiling/{TIMESTAMP}/operation_statistic_{executeCount}.txt`。
- 算子信息，具体路径是 `{DUMP_DIR}/ait_dump/operation_io_tensors/{PID}/operation_tensors_{executeCount}.csv`。
- kernel算子信息，具体路径是 `{DUMP_DIR}/ait_dump/kernel_io_tensors/{PID}/kernel_tensors_{executeCount}.csv`。

##### 模型拓扑信息转onnx可视化模型：

```python
from ait_llm.common.json_fitter import atb_json_to_onnx

model_level = 1   # 可视化模型的节点深度，按需填写，比如填写为1，则表示生成深度为1的可视化模型，不填默认生成最大深度可视化模型
layer_topo_info = "./XXX_layer.json"   # dump出来的layer拓扑信息或者model拓扑信息
atb_json_to_onnx(layer_topo_info, model_level)
```

### 其他结果查看

todo，比如 model信息有什么内容，layer 信息有什么内容， 