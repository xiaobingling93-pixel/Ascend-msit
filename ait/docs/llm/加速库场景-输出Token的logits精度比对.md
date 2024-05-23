# 输出 Token 的 logits 精度比对 - 加速卡推理场景

在加速库推理场景，当存在精度问题时，用户需要排查到底是哪个输出 Token 出现了精度误差。以下介绍如何对输出 Token 进行精度比对。

## 对比思路

使用方法分为三步：

1. dump torch 模型 token 的输出 logits
2. dump 加速库 token 的输出 logits
3. 将上面两个步骤生成的 dump 文件，输入到 ait llm compare 中完成比对

## 第一步：dump torch 模型 token 的输出 logits

使用[在线图例 DUMP 功能](/ait/docs/llm/工具-DUMP在线推理数据使用说明.md)。将模型最后的输出的 logits dump 下来。

```python
from ait_llm import DumpConfig, register_hook #在模型py文件中文件开头导入DumpConfig和register_hook

# dump_last_logits=True 参数表示要 dump 输出 logits
# dump_path="./torch_dump" 参数指定 dump 保存路径
dump_config = DumpConfig(dump_last_logits=True, dump_path="./torch_dump")
register_hook(model, dump_config)  # model是要dump中间tensor的模型实例，在模型初始化后添加代码

```

> dump 数据路径：
>
> - logits 信息落盘位置：在`{DUMP_DIR}/{cuda|cpu}{device_id}_{PID}/{TID}`目录下，会输出所有 token 的 logits
>   注：`device_id`为设备号；`PID`为进程号；`TID`为`token_id`

## 第二步：dump 加速库 token 的输出 logits

使用[加速库 Dump 功能](/ait/docs/llm/工具-DUMP加速库数据使用说明.md)。需要 Dump 两次：

1. 先 dump 网络模型，获取到网络最后一层的编号
2. 再通过编号，指定 Dump 最后一层的输出

### 使用方式

1. 先 dump 网络模型，获取到网络最后一层的编号
  ```bash 
    ait llm dump --exec "bash run.sh" --type layer
  ```

  > - `--exec`参数是指定拉起执行大模型推理脚本的命令
  > - `--type layer`参数是指定 dump 模型的 layer 层信息
  > - layer 信息落盘位置：在`{DUMP_DIR}/ait_dump/layer/{PID}/`目录下，可以通过layer信息，获取到最大的编号，编号最大的既是模型最后一层

2. 通过编号，指定 Dump 最后一层的输出
  ```bash 
    ait llm dump --exec "bash run.sh" --type model tensor --ids 上一步骤获取的编号 -er 0,1000 -child False -stp 1
  ```

  > - `--exec`参数是指定拉起执行大模型推理脚本的命令
  > - `--type model tensor`参数是指定 dump 模型的 model 拓扑信息，以及最后一层输出tensor
  > - `-ids 上一步骤获取的编号`参数是指定 dump 的最后一层
  > - `-er 0,1000`参数指定dump的token范围，可以根据实际情况自行决定
  > - `-child False`参数指定不dump子操作的tensor，减少所需空间
  > - `-stp 1`参数指定只dump算子的输出
  > - tensor 信息落盘位置：tensor 信息落盘位置：在`{DUMP_DIR}/{PID}_npu{device_id}/{TID}`目录下


## 第三步：将上面两个步骤生成的 dump 文件，输入到 ait llm compare 中完成比对

使用[自动比对功能](/ait/docs/llm/工具-自动比对功能使用说明.md)。比对标杆数据和加速库数据。

### 使用方式

```shell
ait llm compare -gp {DUMP_DIR}/{cuda|cpu}{device_id}_{PID}/ -mp ait_dump/tensors/{device_id}_{PID}/ -cl token
```

ait llm compare 提供有精度问题的数据与标杆数据之间的比对能力。

> - --golden-path 参数为第一步中 `torch_tensor`所在目录 `{DUMP_DIR}/{PID}_npu{device_id}/`
> - --my-path 参数为第二步中 `atb_tensor`所在目录 `{DUMP_DIR}/ait_dump/tensors/{device_id}_{PID}/`
> - -cl token 参数指定比对的token 数据。
> - 完成比对后会在 `output_dir`下生成一个 `ait_cmp_report_{TIMESTAMP}.csv`，保存比对的最终结果。
> - csv报告查看参考[精度比对结果参数说明](/ait/docs/llm/精度比对结果参数说明.md)