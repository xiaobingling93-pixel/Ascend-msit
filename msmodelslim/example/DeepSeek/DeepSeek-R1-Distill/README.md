# DeepSeek R1 Distill 量化案例


## 此模型仓已适配的模型版本

- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/main)
- [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tree/main)
- [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main)
- [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/tree/main)
- [DeepSeek-R1-Distill-LLaMA-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/tree/main)
- [DeepSeek-R1-Distill-LLaMA-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/tree/main)

## 环境配置


- 使用 MIndIE1.0版本 [官方镜像](https://gitee.com/ascend/ascend-docker-image/tree/dev/mindie#%E5%90%AF%E5%8A%A8%E5%AE%B9%E5%99%A8)，如1.0.0-800I-A2-py311-openeuler24.03-lts

## 量化
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

#### DeepSeek-R1-Distill-Llama 量化
##### DeepSeek-R1-Distill-Llama-8B w8a8量化
  ```shell
  cd msit/msmodelslim/example/Llama
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --anti_method m1
  ```

##### DeepSeek-R1-Distill-Llama-70B w8a8量化
  ```shell
  cd msit/msmodelslim/example/Llama
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m4 --act_method 3 
  ```

#### DeepSeek-R1-Distill-Qwen 量化
##### DeepSeek-R1-Distill-Qwen-1.5B w8a8量化
  ```shell
  cd msit/msmodelslim/example/Qwen
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu
  ```

##### DeepSeek-R1-Distill-Qwen-7B w8a8量化
  ```shell
  cd msit/msmodelslim/example/Qwen
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu
  ```

##### DeepSeek-R1-Distill-Qwen-14B w8a8量化
  ```shell
  cd msit/msmodelslim/example/Qwen
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu
  ```

##### DeepSeek-R1-Distill-Qwen-32B w8a8量化
  ```shell
  cd msit/msmodelslim/example/Qwen
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu
  ```