# 导入所需的库
import json
import os

import logging

from safetensors.torch import load_file  # 用于加载safetensors格式的文件
from modelslim.pytorch.weight_compression import CompressConfig, Compressor  # 用于模型权重压缩的配置和压缩器

# 准备待压缩权重文件和相关压缩配置，请根据实际情况进行修改
# 定义权重文件所在的目录路径
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/weight_compression_safetensor"
# 待压缩权重文件的路径
weight_path = os.path.join(LOAD_PATH, "quant_model_weight_w8a8s.safetensors")
# 待压缩权重文件的描述文件的路径
json_path = os.path.join(LOAD_PATH, "quant_model_description_w8a8s.json")

# 使用CompressConfig接口，配置压缩参数，并返回配置实例
# do_pseudo_sparse: 是否使用伪稀疏压缩
# sparse_ratio: 稀疏比率
# is_debug: 是否开启调试模式
# record_detail_root: 记录压缩详情的根目录
# multiprocess_num: 多进程数量
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True,
                                 record_detail_root=f"{os.environ['PROJECT_PATH']}/output/weight_compression_safetensor", multiprocess_num=8)

# 加载待压缩的权重文件
sparse_weight = load_file(weight_path)
# 加载权重文件的描述文件
with open(json_path, 'r') as f:
    quant_model_description = json.load(f)

# 使用Compressor接口，输入加载的压缩配置和待压缩权重文件
compressor = Compressor(compress_config, weight=sparse_weight, quant_model_description=quant_model_description)
# 执行压缩操作
compress_weight, compress_index, compress_info = compressor.run()

# 使用export_safetensors()接口，保存压缩后的结果文件
# 这里需要指定保存压缩结果的目录路径，以及压缩后的safetensors文件名和json文件名
compressor.export_safetensors(f"{os.environ['PROJECT_PATH']}/output/weight_compression_safetensor", safetensors_name=None, json_name=None)
logging.info("Weight Compression safetensor success!")