# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import argparse
import json
from typing import List

import numpy as np
import torch


from safetensors.torch import load_file, save_file, safe_open


QUANT_TYPE = ['W4A16', 'W8A16']
ORDINAL_PACK_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]
AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
GPTQ_PACK_ORDER = [7, 6, 5, 4, 3, 2, 1, 0]


def awq_pack(iweight: torch.Tensor, w_bit:4, direction: str = "column"):
    storage_bits = 32
    pack_num = storage_bits // w_bit
    shifts = torch.arange(0, storage_bits, w_bit, device=iweight.device)

    iweight = iweight.to(torch.int8)
    iweight = torch.bitwise_and(iweight, 0x0F)  # eventually correct overflow

    if direction == "column":
        iweight = iweight.view(-1, iweight.shape[1] // pack_num, pack_num)
        qmatrix = torch.bitwise_left_shift(iweight, shifts[None, None, :]).sum(dim=-1)

    elif direction == "row":
        iweight = iweight.view(iweight.shape[0] // pack_num, pack_num, -1)
        qmatrix = torch.bitwise_left_shift(iweight, shifts[None, :, None]).sum(dim=1)

    qmatrix = qmatrix.to(torch.int32)
    return qmatrix


def apply_order(
    iweight: torch.Tensor,
    w_bit: int,
    direction: str = "column",
    order: List[int] = None,
):
    storage_bits = 32
    pack_num = storage_bits // w_bit
    if direction == "column":
        iweight = iweight.view(-1, pack_num)[:, order].view(iweight.shape)
    elif direction == "row":
        iweight = iweight.view(pack_num, -1)[order, :].view(iweight.shape)
    return iweight


def gptq_qweight_pack(iweight:torch.Tensor, w_bit:4):
    i = 0
    row = 0
    storage_bits = 32
    iweight = iweight.numpy().astype(np.uint32)
    qweight = np.zeros((iweight.shape[0] // storage_bits * w_bit, iweight.shape[1]), dtype=np.uint32)
    while row < qweight.shape[0]:
        if w_bit in [4, 8]:
            for j in range(i, i + (32 // w_bit)):
                qweight[row] |= iweight[j] << (w_bit * (j - i))
            i += 32 // w_bit
            row += 1
    qweight = qweight.astype(np.int32)
    qweight = torch.from_numpy(qweight)
    return qweight


def gptq_qzeros_pack(zeros:torch.Tensor, w_bit:4):
    i = 0
    col = 0
    storage_bits = 32
    if zeros.dtype == torch.bfloat16:
        zeros = zeros.to(torch.float16)
    zeros = zeros.numpy().astype(np.uint32)
    qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // storage_bits * w_bit), dtype=np.uint32)
    while col < qzeros.shape[1]:
        if w_bit in [2, 4, 8]:
            for j in range(i, i + (32 // w_bit)):
                qzeros[:, col] |= zeros[:, j] << (w_bit * (j - i))
            i += 32 // w_bit
            col += 1
            
    qzeros = qzeros.astype(np.int32)
    qzeros = torch.from_numpy(qzeros)
    return qzeros  


def convert_ms_to_vllm(targer_tool, w_bit, weight_dict, json_dict):
    vllm_weight_dict = {}
    for name, quant_type in json_dict.items():
        if name in weight_dict.keys():
            tensor = weight_dict[name]
            if quant_type in QUANT_TYPE:
                order = AWQ_PACK_ORDER
                direction = 'column'
                if name.endswith('.weight') and 'module.weight' not in name:
                    vllm_name = '.'.join(name.split('.')[:-1]) + '.qweight'
                    tensor = tensor.t().contiguous()
                    tensor.add_(8)
                    if (targer_tool == 'awq'):
                        iweights = apply_order(tensor, w_bit, direction, order)
                        qweight = awq_pack(iweights, w_bit)
                    else:
                        qweight = gptq_qweight_pack(tensor, w_bit)
                    vllm_weight_dict[vllm_name] = qweight
                
                elif name.endswith('.weight_scale'):
                    vllm_name = '.'.join(name.split('.')[:-1]) + '.scales'
                    tensor = tensor.t().contiguous()
                    vllm_weight_dict[vllm_name] = tensor

                elif name.endswith('.weight_offset'):
                    vllm_name = '.'.join(name.split('.')[:-1]) + '.qzeros'
                    tensor = tensor.t().contiguous()
                    tensor.add_(8)
                    if (targer_tool == 'awq'):
                        izeros = apply_order(tensor, w_bit, direction, order)
                        qzeros = awq_pack(izeros, w_bit, direction)
                    else:
                        qzeros = gptq_qzeros_pack(tensor, w_bit)
                    vllm_weight_dict[vllm_name] = qzeros
                
                elif 'module.weight' in name and 'model.norm' not in name:
                    vllm_name = name.replace('module.weight', 'weight')
                    vllm_weight_dict[vllm_name] = tensor
                elif 'model.norm.module.bias' in name or 'model.norm.module.weight' in name:
                    pass
                else:
                    vllm_weight_dict[name] = tensor
            else:
                vllm_weight_dict[name] = tensor
    return vllm_weight_dict
    

def load_json_info(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        quant_info = json.load(f)

    return quant_info
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Quantied safetensors file path")
    parser.add_argument("--json", type=str, default=None, help="Quantied description file path")
    parser.add_argument("--save_path", type=str, default='res.safetensors', help="The path to save converted quant weights")
    parser.add_argument("--w_bit", type=int, default=4, help="Quantied weight bits")
    parser.add_argument("--target_tool", type=str, default="awq", help="target tool, value include awq and gptq")
    args = parser.parse_args()

    save_path = args.save_path
    w_bit = args.w_bit
    quant_tool = args.target_tool

    tensor_info = load_file(args.model)
    json_info = load_json_info(args.json)

    vllm_weight = convert_ms_to_vllm(quant_tool, w_bit, weight_dict=tensor_info, json_dict=json_info)
    save_file(vllm_weight, args.save_path)
