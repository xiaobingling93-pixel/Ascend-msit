import os

import torch
from ait_llm.common.log import logger
from ait_llm.compare.cmp_utils import BasicDataInfo, fill_row_data
from ait_llm.compare.cmp_utils import read_data


def multi_block_cmp(atb_nodes, torch_nodes, my_root_node, atb_tensor_path, torch_tensor_path):
    compared_result = []
    for atb_node, torch_node in zip(atb_nodes, torch_nodes):
        if atb_node.op_type == "LinearOperation" and not atb_node.op_param.get("hasBias"):
            next_sibling_node = my_root_node.get_next_sibling_node(atb_node)
            # 当有些算子如ParallelLinearBaseV2，是将w*x+b的操作拆分成两个算子，linear+add，而torch中使用一个算子Linear实现，
            # 因此add node的输出映射的是torch中Linear的输出
            if next_sibling_node and next_sibling_node.op_type == "ElewiseOperation" \
                    and next_sibling_node.op_param.get('elewiseType') == 8:
                atb_node = next_sibling_node
        atb_node_tensor_path = atb_node.tensor_path
        torch_node_tensor_path = torch_node.tensor_path
        atb_multi_block_tensor_path_name = os.path.basename(os.path.abspath(os.path.join(atb_tensor_path, "..")))
        atb_multi_block_tensor_path = os.path.abspath(os.path.join(atb_tensor_path, "..", ".."))
        atb_multi_block_tensor_paths = os.listdir(atb_multi_block_tensor_path)
        torch_multi_block_tensor_path_name = os.path.basename(
            os.path.abspath(os.path.join(torch_tensor_path, "..")))
        torch_multi_block_tensor_path = os.path.abspath(os.path.join(torch_node_tensor_path, "..", ".."))
        torch_multi_block_tensor_paths = os.listdir(torch_multi_block_tensor_path)
        my_tensor_path = os.path.abspath(os.path.join(atb_node_tensor_path, "after", "outtensor0.bin"))
        golden_tensor_path = os.path.abspath(os.path.join(torch_node_tensor_path, "output.pth"))
        if not os.path.exists(my_tensor_path) or not os.path.exists(golden_tensor_path):
            msg = f"golden tensor path: {golden_tensor_path} or my_tensor_path: {my_tensor_path} is not exist."
            logger.debug(msg)
            continue
        # 1. load torch_data and atb_data
        my_tensor_data = read_data(my_tensor_path)
        golden_tensor_data = read_data(golden_tensor_path)
        # 2. concat multiples block tensor
        dim_atb = get_cat_dim(torch_multi_block_tensor_paths, atb_multi_block_tensor_paths, my_tensor_data,
                              golden_tensor_data)
        my_tensor_data = concat_tensor_data(my_tensor_data, atb_multi_block_tensor_path_name,
                                            atb_multi_block_tensor_paths, my_tensor_path, dim_atb)
        dim_torch = get_cat_dim(atb_multi_block_tensor_paths, torch_multi_block_tensor_paths, golden_tensor_data,
                                my_tensor_data)
        golden_tensor_data = concat_tensor_data(golden_tensor_data, torch_multi_block_tensor_path_name,
                                                torch_multi_block_tensor_paths, golden_tensor_path, dim_torch)
        # 3. compare tensor
        compare_row_data(compared_result, golden_tensor_path, my_tensor_path, golden_tensor_data, my_tensor_data)

    return compared_result


def get_cat_dim(atb_multi_block_tensor_paths, torch_multi_block_tensor_paths, golden_tensor_data, my_tensor_data):
    for dim, (golden_size, my_size) in enumerate(zip(golden_tensor_data.shape, my_tensor_data.shape)):
        multi_golden_size = golden_size * len(torch_multi_block_tensor_paths)
        multi_my_size = my_size * len(atb_multi_block_tensor_paths)
        if multi_golden_size == multi_my_size and golden_size != my_size:
            return dim
        if multi_golden_size == my_size:
            return dim
    return -1


def concat_tensor_data(tensor_data, multi_block_tensor_path_name,
                       multi_block_tensor_paths, tensor_path, dim):
    for tensor_path_name in multi_block_tensor_paths:
        if tensor_path_name != multi_block_tensor_path_name:
            tensor_path = tensor_path.replace(multi_block_tensor_path_name, tensor_path_name)
            if dim == -1 or not os.path.exists(tensor_path):
                continue
            tensor_data = torch.cat((tensor_data, read_data(tensor_path)), dim)

    return tensor_data


def compare_row_data(compared_result, golden_tensor_path, my_tensor_path, golden_tensor_data, my_tensor_data):
    data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, data_id=0)
    row_data = fill_row_data(data_info, my_tensor_data, golden_tensor_data)
    compared_result.append(row_data)






