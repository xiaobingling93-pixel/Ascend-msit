import os
import torch
from ait_llm.compare.cmp_utils import compare_data, read_data
from ait_llm.common.log import logger
from ait_llm.compare.cmp_utils import BasicDataInfo, fill_row_data, save_compare_reault_to_csv
from ait.components.llm.ait_llm.common.constant import CMP_FAIL_REASON
from ait.components.llm.ait_llm.compare.cmp_utils import set_tensor_basic_info_in_row_data


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
        atb_multi_block_tensor_path_name = os.path.basename(os.path.abspath(os.path.join(atb_tensor_path, "../")))
        atb_multi_block_tensor_path = os.path.abspath(os.path.join(atb_tensor_path, "../../"))
        atb_multi_block_tensor_paths = os.listdir(atb_multi_block_tensor_path)
        torch_multi_block_tensor_path_name = os.path.basename(
            os.path.abspath(os.path.join(torch_tensor_path, "../")))
        torch_multi_block_tensor_path = os.path.abspath(os.path.join(torch_node_tensor_path, "../../"))
        torch_multi_block_tensor_paths = os.listdir(torch_multi_block_tensor_path)
        my_tensor_path = os.path.join(atb_node_tensor_path, "after", "outtensor0.bin")
        golden_tensor_path = os.path.join(torch_node_tensor_path, "output.pth")
        if not os.path.exists(my_tensor_path) or not os.path.exists(golden_tensor_path):
            msg = f"golden tensor path: {golden_tensor_path} or my_tensor_path: {my_tensor_path} is not exist."
            logger.debug(msg)
            continue
        # atb第一张卡tensor数据
        my_tensor_data = read_data(my_tensor_path)
        # torch第一张卡tensor数据
        golden_tensor_data = read_data(golden_tensor_path)
        # atb单卡-torch单卡
        if len(atb_multi_block_tensor_paths) == 1 and len(torch_multi_block_tensor_paths) == 1:
            data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, data_id=0)
            row_data = fill_row_data(data_info)
            compared_result.append(row_data)
        # atb单卡-torch多卡
        if len(atb_multi_block_tensor_paths) == 1 and len(torch_multi_block_tensor_paths) > 1:
            golden_tensor_data, golden_tensor_path, torch_node_tensor_path = single_atb_to_multi_torch(compared_result,
                                                                                                       golden_tensor_data,
                                                                                                       golden_tensor_path,
                                                                                                       my_tensor_data,
                                                                                                       my_tensor_path,
                                                                                                       torch_multi_block_tensor_path_name,
                                                                                                       torch_multi_block_tensor_paths,
                                                                                                       torch_node_tensor_path)
        # atb多卡-torch单卡
        if len(atb_multi_block_tensor_paths) > 1 and len(torch_multi_block_tensor_paths) == 1:
            atb_node_tensor_path, my_tensor_data, my_tensor_path = single_torch_to_multi_atb(
                atb_multi_block_tensor_path_name, atb_multi_block_tensor_paths, atb_node_tensor_path, compared_result,
                golden_tensor_data, golden_tensor_path, my_tensor_data, my_tensor_path)
        # atb多卡-torch多卡
        if len(atb_multi_block_tensor_paths) > 1 and len(torch_multi_block_tensor_paths) > 1:
            if my_tensor_data.shape == golden_tensor_data.shape:
                get_compare_result(compared_result, golden_tensor_data, golden_tensor_path, my_tensor_data,
                                   my_tensor_path)
            else:
                compare_multi_block_tensor(atb_multi_block_tensor_path_name, atb_multi_block_tensor_paths,
                                           atb_node_tensor_path,
                                           compared_result, golden_tensor_data, my_tensor_data,
                                           torch_multi_block_tensor_path_name,
                                           torch_multi_block_tensor_paths, torch_node_tensor_path)

    return compared_result


def single_torch_to_multi_atb(atb_multi_block_tensor_path_name, atb_multi_block_tensor_paths, atb_node_tensor_path,
                              compared_result, golden_tensor_data, golden_tensor_path, my_tensor_data, my_tensor_path):
    dim = -1
    for i, size in enumerate(golden_tensor_data.shape):
        if size == my_tensor_data.shape[i] * len(atb_multi_block_tensor_paths):
            dim = i
    for tensor_path_name in atb_multi_block_tensor_paths:
        if tensor_path_name != atb_multi_block_tensor_path_name:
            atb_node_tensor_path = atb_node_tensor_path.replace(atb_multi_block_tensor_path_name
                                                                , tensor_path_name)
            my_tensor_path = os.path.join(atb_node_tensor_path, "after", "outtensor0.bin")
            if not os.path.exists(my_tensor_path):
                continue
            if dim != -1:
                my_tensor_data = torch.cat((my_tensor_data, read_data(my_tensor_path)), dim)
    get_compare_result(compared_result, golden_tensor_data, golden_tensor_path, my_tensor_data,
                       my_tensor_path)
    return atb_node_tensor_path, my_tensor_data, my_tensor_path


def single_atb_to_multi_torch(compared_result, golden_tensor_data, golden_tensor_path, my_tensor_data, my_tensor_path,
                              torch_multi_block_tensor_path_name, torch_multi_block_tensor_paths,
                              torch_node_tensor_path):
    dim = -1
    for i, size in enumerate(my_tensor_data.shape):
        if size == golden_tensor_data.shape[i] * len(torch_multi_block_tensor_paths):
            dim = i
    for tensor_path_name in torch_multi_block_tensor_paths:
        if tensor_path_name != torch_multi_block_tensor_path_name:
            torch_node_tensor_path = torch_node_tensor_path.replace(torch_multi_block_tensor_path_name
                                                                    , tensor_path_name)
            golden_tensor_path = os.path.join(torch_node_tensor_path, "output.pth")
            if not os.path.exists(golden_tensor_path):
                continue
            if dim != -1:
                golden_tensor_data = torch.cat((golden_tensor_data, read_data(golden_tensor_path)), dim)
    get_compare_result(compared_result, golden_tensor_data, golden_tensor_path, my_tensor_data,
                       my_tensor_path)
    return golden_tensor_data, golden_tensor_path, torch_node_tensor_path


def fill_compare_data(data_info, golden_tensor_data, my_tensor_data):
    golden_data_path, my_data_path = data_info.golden_data_path, data_info.my_data_path
    logger.debug(f"[fill_row_data], golden_data_path: {golden_data_path}, my_data_path: {my_data_path}")
    row_data = data_info.to_dict()
    if not os.path.isfile(golden_data_path):
        row_data[CMP_FAIL_REASON] = f"golden_data_path: {golden_data_path} is not a file."
        return row_data
    if not os.path.isfile(my_data_path):
        row_data[CMP_FAIL_REASON] = f"my_data_path: {my_data_path} is not a file."
        return row_data
    row_data.update(compare_data(golden_tensor_data, my_tensor_data))
    row_data.update(set_tensor_basic_info_in_row_data(golden_tensor_data, my_tensor_data))

    return row_data


def get_compare_result(compared_result, golden_tensor_data, golden_tensor_path, my_tensor_data, my_tensor_path):
    data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, data_id=0)
    compared_result.append(fill_compare_data(data_info, golden_tensor_data, my_tensor_data))


def compare_multi_block_tensor(atb_multi_block_tensor_path_name, atb_multi_block_tensor_paths, atb_node_tensor_path,
                               compared_result,
                               golden_tensor_data, my_tensor_data, torch_multi_block_tensor_path_name,
                               torch_multi_block_tensor_paths,
                               torch_node_tensor_path):
    # torch的切割维度
    dim_torch = get_cat_dim(atb_multi_block_tensor_paths, golden_tensor_data, my_tensor_data,
                            torch_multi_block_tensor_paths)
    # atb的切割维度
    dim_atb = get_cat_dim(torch_multi_block_tensor_paths, my_tensor_data, golden_tensor_data
                          , atb_multi_block_tensor_paths)

    my_tensor_data, my_tensor_path = concat_tensor_data(atb_multi_block_tensor_path_name,
                                                        atb_multi_block_tensor_paths,
                                                        atb_node_tensor_path, dim_atb,
                                                        my_tensor_data, ["after", "outtensor0.bin"])
    golden_tensor_data, golden_tensor_path = concat_tensor_data(torch_multi_block_tensor_path_name,
                                                                torch_multi_block_tensor_paths,
                                                                torch_node_tensor_path,
                                                                dim_torch, golden_tensor_data,
                                                                ["output.pth"])
    data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, data_id=0)
    compared_result.append(fill_compare_data(data_info, golden_tensor_data, my_tensor_data))


def concat_tensor_data(multi_block_tensor_path_name, multi_block_tensor_paths, node_tensor_path, dim,
                       tensor_data, bin_path_list):
    for tensor_path_name in multi_block_tensor_paths:
        if tensor_path_name != multi_block_tensor_path_name:
            node_tensor_path = node_tensor_path.replace(multi_block_tensor_path_name, tensor_path_name)
            for bin_path in bin_path_list:
                node_tensor_path = os.path.join(node_tensor_path, bin_path)
            if dim == -1 or not os.path.exists(node_tensor_path):
                continue
            tensor_data = torch.cat((tensor_data, read_data(node_tensor_path)), dim)
    return tensor_data, node_tensor_path


def get_cat_dim(atb_multi_block_tensor_paths, golden_tensor_data, my_tensor_data,
                torch_multi_block_tensor_paths):
    for dim, (golden_size, my_size) in enumerate(zip(golden_tensor_data.shape, my_tensor_data.shape)):
        multi_golden_size = golden_size * len(torch_multi_block_tensor_paths)
        multi_my_size = my_size * len(atb_multi_block_tensor_paths)
        if multi_golden_size == multi_my_size and golden_size != my_size:
            return dim
        if multi_golden_size == my_size:
            return dim
    return -1
