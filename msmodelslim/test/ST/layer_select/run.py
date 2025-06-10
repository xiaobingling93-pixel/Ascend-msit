import os
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
 
SEQ_LEN_OUT = 32
 
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=LOAD_PATH,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=LOAD_PATH,
                                             torch_dtype=torch.float32, trust_remote_code=True).cpu()
 
'''
# for 中文模型
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
'''
 
calib_list = ["Where is the capital of China?",
              "Please make a poem:",
              "I want to learn python, how should I learn it?",
              "Please help me write a job report on large model inference optimization:",
              "What are the most worth visiting scenic spots in China?"]
 
 
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset
 
 
dataset_calib = get_calib_dataset(tokenizer, calib_list)
 
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector
ls = LayerSelector(model, range_method='quantile')
ls.run(dataset_calib)
layers = ls.select_layers_by_threshold(1)
print("layer select by threshold:", layers)
layers = ls.select_layers_by_disable_level(2)
print("layer select by disable level:", layers)