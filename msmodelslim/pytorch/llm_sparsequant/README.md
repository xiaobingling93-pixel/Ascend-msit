## LLM Sparse Quant Tool
LLM Sparse Quant Tool是基于LLM PTQ工具的大模型稀疏量化工具

## 使用教程

教程分为3步：

1. 准备校准数据

2. 定义校准config

3. 执行PTQ量化校准 + 存储量化参数用于部署

 

### Step1.  准备校准数据
目前，支持的模型为ChatGLM2-6B，针对此模型进行了稀疏量化的优化，下述的教程以此模型为主。

目前由于模型的局限性，暂时使用的是C-Eval的一小部分验证集作为校准数据集，如下代码所示。其中`entry`为CEval数据集的`mao_zedong_thought.jsonl`的验证集，这个数据集便是我们的校准数据
```python
    fp_16_path = "path/to/your/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=fp16_path,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=fp16_path,
                                      torch_dtype=torch.float32, trust_remote_code=True)
    
    choices = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]
    
    def build_prompt(text):
        return "[Round {}]\n\n问：{}\n\n答：".format(1, text)
    
    extraction_prompt = '综上所述，ABCD中正确的选项是：'
    
    def get_dataset(bs=1):
        with torch.no_grad():
            dataset_all = []          
            entry = "PathToCEvalDataset/CEval/val/Social_Science/mao_zedong_thought.jsonl"
            dataset_cur = []
            dataset = []
            with open(entry, encoding='utf-8') as file:
                for line in file:
                    dataset.append(json.loads(line))
            correct = 0
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs)
            for batch in tqdm(dataloader):
                texts = batch["inputs_pretokenized"]
                queries = [build_prompt(query) for query in texts]
                inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(
                    'cpu')
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
                intermediate_outputs = []
                for idx in range(len(outputs)):
                    output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                    response = tokenizer.decode(output)
                    print('response: ', response)
                    intermediate_outputs.append(response)
                answer_texts = [text + intermediate + "\n" + extraction_prompt for text, intermediate in
                                zip(texts, intermediate_outputs)]
                input_tokens = [build_prompt(answer_text) for answer_text in answer_texts]
                inputs = tokenizer(input_tokens, padding=True, return_tensors="pt", truncation=True,
                                   max_length=2048).to('cpu')
                dataset_tmp = [inputs.data['input_ids'], inputs.data['position_ids'], inputs.data['attention_mask'], None, None, None, None, None,
                               None, None, True]             
                dataset_cur.append(dataset_tmp)          
            dataset_all.extend(dataset_cur)
        return dataset_all

    dataset_calib = get_dataset()
    print('len of calib_dataset: ', len(dataset_calib))
    dataset_calib = dataset_calib[:]
```


### Step2.  定义校准config
```python
    w_bit = 4
    fraction = 0.007
    powerquant = False
    mm_tensor = True
    
    quant_config = QuantConfig(w_bit=w_bit,
                               disable_names=['transformer.encoder.layers.0.mlp.dense_4h_to_h', 'transformer.output_layer'],
                               dev_type='cpu',
                               act_method=3,
                               pr=2.0,
                               fraction=fraction,
                               nonuniform=powerquant,
                               mm_tensor=mm_tensor,
                               co_sparse=True)
```
### Step3.  执行PTQ量化校准 + 存储量化参数用于部署
```python
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # 内部回退两层
    calibrator.run(int_infer=False)
    calibrator.save('save_path') #存储量化参数用于部署，在存储量化参数过程中，存在反序列化风险，已通过将保存的量化结果文件夹权限设置为750，将量化结果文件权限设置为400来消减该风险
```

## 工具参数简介
### 一、LLM稀疏量化介绍
#### 稀疏量化config(`QuantConfig`):
+ pr:量化正则百分比,目前的模型默认为2.0，即默认不开启量化正则的效果(根据内部实验，默认值即可)
+ fraction:与精度相关的参数，torch1.X版本建议`0.007`，2.X版本建议`0.011`
+ mm_tensor:权重量化方式。True: per-tensor量化； False: per-channel量化，在大模型场景，使用per-channel,在torch2.X的版本中建议关闭
+ co_sparse:是否开启稀疏量化功能，如使用此工具需要打开此选项

其余参数可以参考LLM PTQ的`QuantConfig`

#### Calibrator类参数传入：
+ model：用于量化的模型
+ quant_config：量化config，使用llm_ptq_tools中QuantConfig构建
+ calib_data：用于LLM大模型量化校准的数据
+ disable_level：自动回退等级，从L0——L5可选，L0代表不回退，L1——L5对应等级逐渐增大，在模型精度损失大可以适当提升等级，其中在ChatGLM v2中使用了L2等级，自动去除了一个最大层的量化
