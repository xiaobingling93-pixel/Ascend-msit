
### 1. GE开启融合（默认） Dump 案例
  ```py
  import os, numpy as np
  import torch, torch_npu, torchair as tng, torchvision
  from ait_llm.dump import torchair_dump

  target_dtype = torch.float16
  model = torchvision.models.resnet50(pretrained=True).eval().to(target_dtype).npu()
  if not os.path.exists('aa_224_224.npy'):
      np.save('aa_224_224.npy', np.random.uniform(size=[1,3,224,224]))
  aa = torch.from_numpy(np.load('aa_224_224.npy')).to(target_dtype).npu()
  config = torchair_dump.get_ge_dump_config(dump_path="dump")
  npu_backend = tng.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  with torch.no_grad():
      try:
          print(model(aa).shape)
      except:
          pass
  ```
- 其中get_ge_dump_config 参数列表

  | 参数名                | 参数描述                                                  | 是否必选                |
  |--------------------|-------------------------------------------------------|---------------------|
  | dump_path          | dump数据的存放路径                                           | 是                   |
  | dump_model         | data dump模式，用于指定dump算子输入还是输出数据                        | 否                   |
  | fusion_switch_file | 是否关闭融合dump功能                                          | 否(默认为false，开启融合)    | 
  | dump_token         | 指定token进行dump, 格式："1,2-5", 代表dump第1、2、3、4、5个token数据   | 否(默认为None，dump全量数据) |, 
  | dump_layer         | 指定layer进行dump, 格式："Add,Conv_1", 代表dump Add和Conv_1两层数据 | 否(默认为None，dump全量数据) | 

### 2. FX Dump 案例
  ```py
  import os, numpy as np
  import torch, torch_npu, torchair as tng, torchvision
  from ait_llm.dump import torchair_dump

  target_dtype = torch.float16
  model = torchvision.models.resnet50(pretrained=True).eval().to(target_dtype).npu()
  if not os.path.exists('aa_224_224.npy'):
      np.save('aa_224_224.npy', np.random.uniform(size=[1,3,224,224]))
  aa = torch.from_numpy(np.load('aa_224_224.npy')).to(target_dtype).npu()
  config = torchair_dump.get_fx_dump_config()
  npu_backend = tng.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  with torch.no_grad():
      try:
          print(model(aa).shape)
      except:
          pass
  ```

### 3. GE关闭融合 Dump 案例
  ```py
  import os, numpy as np
  import torch, torch_npu, torchair as tng, torchvision
  from ait_llm.dump import torchair_dump

  target_dtype = torch.float16
  model = torchvision.models.resnet50(pretrained=True).eval().to(target_dtype).npu()
  if not os.path.exists('aa_224_224.npy'):
      np.save('aa_224_224.npy', np.random.uniform(size=[1,3,224,224]))
  aa = torch.from_numpy(np.load('aa_224_224.npy')).to(target_dtype).npu()
  config = torchair_dump.get_ge_dump_config(dump_path="dump", fusion_switch_file='./fusion_switch.json')
  npu_backend = tng.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  with torch.no_grad():
      try:
          print(model(aa).shape)
      except:
          pass
  ```
  新建fusion_switch.json文件，内容如下：
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