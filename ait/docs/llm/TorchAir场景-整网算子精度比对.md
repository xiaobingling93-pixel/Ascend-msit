
# 2. 基于torch图模式（torchair）推理场景

## 2.1 GE dump 数据与 FX dump 数据精度比对

### 1）Dump 数据

- **GE 模式 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from ait_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

- **FX 模式 dump 数据** 添加 `get_fx_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from ait_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_fx_dump_config()  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  - 输出路径与 torchair 版本相关，新版本中为当前文件夹下的 `data_dump/{token_id}/gm_{time stamp}_dump`，老版本中为 `gm_{time stamp}_dump`
  - **其中 `{token_id}` 是从 1 开始的，相对于 GE 模式是从 0 开始的，比对时会将 FX 模式的 token_id 减 1**

### 2）Compare 精度比对

  - 执行 `ait llm compare --my-path [GE dump data] --golden-path [FX dump data]`，输出比对结果 csv 文件

    ```sh
    ait llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path data_dump
    ```

***

## 2.2 融合的 GE dump 数据与关闭融合的 GE dump 数据精度比对

### 1）Dump 数据

- **GE 模式 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from ait_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

- **GE 模式关闭融合 dump 数据** 添加 `get_ge_dump_config`，指定 `fusion_switch_file` 文件，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from ait_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump", fusion_switch_file="fusion_switch.json")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  参考的 `fusion_switch.json` 文件

  ```json
  {
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
      },
      "UBFusion": {
        "ALL": "off"
      }
    }
  }
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

### 2）Compare 比对

- 执行 `ait llm compare --my-path [GE dump data] --golden-path [fusion off GE dump data]`，输出比对结果 csv 文件

  ```sh
  ait llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path {dump_path}/dump_{time_stamp}
  ```

## 结果查看

参考[精度比对结果参数说明](/ait/docs/llm/精度比对结果参数说明.md)

***

## 2.3 将 dump 数据转化为指定信息以压缩数据量
- dump 过程中生成的数据量可能占用大量磁盘空间，可以在 dump 过程中启用后台进程，将完整的数据提取为指定的信息。以下参考脚本将数据转化为最大最小值，并删除原数据
  ```py
  #!/bin/env python3
  import os
  import time
  import argparse

  surfix = "_min_max"  # Converted data save surfix

  # Define how single data is converted
  def convert_data_to_info(data):
      return [data.min(), data.max()]

  def convert(data_path):
      import numpy as np
      from ait_llm.compare import torchair_acc_cmp

      npz_surfix, npy_surfix = "{}.npz".format(surfix), "{}.npy".format(surfix)
      for cur_path, dirs, files in os.walk(data_path):
          for file in files:
              if file.endswith(npy_surfix):  # already converted FX data
                  continue

              cur = os.path.join(cur_path, file)
              if file.endswith(".npy"):  # FX saved npy data
                  file_name = os.path.splitext(cur)[0]
                  np.save(file_name + surfix, convert_data_to_info(np.load(cur)))
                  os.remove(cur)
                  print("Converted: {} -> {}{}".format(cur, file_name, npy_surfix))
              elif not file.endswith(npz_surfix) and not file.endswith(".txt") and not file.endswith(".swp"):
                  inputs, outputs = torchair_acc_cmp.parse_torchair_dump_data(cur)
                  inputs = [convert_data_to_info(ii) for ii in inputs]
                  outputs = [convert_data_to_info(ii) for ii in outputs]

                  np.savez(cur + npz_surfix, inputs=inputs, outputs=outputs)
                  os.remove(cur)
                  print("Converted: {} -> {}{}".format(cur, cur, npz_surfix))
  
  if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("data_path", help="GE or FX data dump path")
      args = parser.parse_args()
      while True:
          convert(args.data_path)
          time.sleep(0.5)
          print("Waiting...")
  ```
  在 dump 过程中后台执行该脚本，将 dump 数据转化为 info 数据，以减少内存占用
  ```sh
  # 将 ait_ge_dump 下的 GE dump 数据转化为 info
  python3 convert.py ait_ge_dump
  ```
  同时在 FX 或关闭融合的 GE dump 时也执行该脚本，将 dump 数据转化为 info。
  ```sh
  # 将 ait_ge_dump 下的 FX dump 数据转化为 info
  python3 convert.py data_dump
  ```
  最后执行比对
  ```sh
  ait llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path data_dump
  ```
  