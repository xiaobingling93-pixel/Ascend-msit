# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import re

import numpy as np
from ait_llm.common.log import logger
from ait_llm.common.utils import safe_string
from ait_llm.compare.cmp_utils import BasicDataInfo, fill_row_data, save_compare_reault_to_csv

GE_GRAPH_FILE_PREFIX = "dynamo_original_graph_"
FUSION_OP_TYPE = "AutomaticBufferFusionOp"
DUMP_FILE_FILTER_SUFIX = [".txt", ".npy", ".bin"]
IS_MSACCUCMP_PATH_SET = False
GLOBAL_TENSOR_CONVERTER = None
MAX_TOKEN_LEN = 12


def default_tensor_converter(tensor):
    return tensor.data.reshape(tensor.shape)


def set_msaccucmp_path_from_cann():
    global IS_MSACCUCMP_PATH_SET
    global GLOBAL_TENSOR_CONVERTER

    # env TOOLCHAIN_HOME works for both development and product packages.
    cann_path = os.environ.get("TOOLCHAIN_HOME", os.environ.get("ASCEND_TOOLKIT_HOME", ""))
    if not cann_path:
        raise OSError("CANN toolkit in not installed or not set, try installing the latest CANN toolkit.")
    cann_path = safe_string(cann_path)
    cann_path = cann_path.split(":")[0]  # Could be multiple split by :, should use the first one

    msaccucmp_path = os.path.join(cann_path, "tools", "operator_cmp", "compare")
    if not os.path.exists(msaccucmp_path):
        raise OSError(f"{msaccucmp_path} not exists, try installing the latest CANN toolkit.")

    if msaccucmp_path not in sys.path:
        sys.path.append(msaccucmp_path)
    IS_MSACCUCMP_PATH_SET = True
    logger.info(f"Set msaccucmp_path={msaccucmp_path}")

    if GLOBAL_TENSOR_CONVERTER is None:
        from conversion import tensor_conversion

        if hasattr(tensor_conversion, "ConvertSingleTensorFormat"):
            GLOBAL_TENSOR_CONVERTER = tensor_conversion.ConvertSingleTensorFormat()
        else:
            GLOBAL_TENSOR_CONVERTER = default_tensor_converter
            logger.warning("ConvertSingleTensorFormat not found in msaccucmp, connot convert tensor format."
                           " Try installing the latest CANN toolkit."
                           )


def get_torchair_ge_graph_path(my_path):
    if not os.path.isdir(my_path):
        return None

    my_path_depth = len(my_path.split(os.sep))
    for cur_path, dirs, file_names in os.walk(my_path):
        for file_name in file_names:
            if file_name.startswith(GE_GRAPH_FILE_PREFIX) and file_name.endswith(".txt"):
                return os.path.join(cur_path, file_name)
            cur_depth = len(cur_path.split(os.sep)) - my_path_depth
            if cur_depth > 5:  # Avoid going too deep
                break
    return None


def parse_torchair_dump_data(dump_file):
    if dump_file.endswith(".npz"):  # Custom converted data info
        loaded = np.load(dump_file)
        return loaded.get("inputs", []), loaded.get("outputs", [])

    if not IS_MSACCUCMP_PATH_SET:
        set_msaccucmp_path_from_cann()
    from dump_parse.dump_utils import parse_dump_file  # Parser tool from CANN msaccucmp
    from cmp_utils.constant.const_manager import ConstManager

    bin_dump_data = parse_dump_file(dump_file, dump_version=ConstManager.OLD_DUMP_TYPE)
    inputs = [GLOBAL_TENSOR_CONVERTER(input_data) for input_data in bin_dump_data.input_data]
    outputs = [GLOBAL_TENSOR_CONVERTER(output_data) for output_data in bin_dump_data.output_data]
    return inputs, outputs


def get_unique_key(cur_dict, cur_key):
    split_sign, original_cur_key, cur_key_id = "#", cur_key, 0
    while cur_key in cur_dict:
        cur_key_id += 1
        cur_key = f"{original_cur_key}{split_sign}{cur_key_id}"
    return cur_key


def parse_pbtxt_to_dict(pbtxt_path):
    with open(pbtxt_path) as ff:
        contents = ff.read()

    result, cur_dict, superior_dicts, brackets_depth = [], {}, [], 0
    for cur_line in contents.split("\n"):
        cur_line = cur_line.strip()
        if len(cur_line) == 0:
            continue

        if " {" in cur_line:
            if brackets_depth == 0:
                cur_dict = {}
                superior_dicts = []
                result.append(cur_dict)
            cur_key = cur_line.split(" {")[0]
            cur_key = get_unique_key(cur_dict, cur_key)
            cur_dict[cur_key] = {}
            if len(superior_dicts) > brackets_depth:
                superior_dicts[brackets_depth] = cur_dict
            else:
                superior_dicts.append(cur_dict)
            cur_dict = cur_dict[cur_key]
            brackets_depth += 1
        elif ": " in cur_line:
            cur_key, cur_value = cur_line.split(": ")
            cur_key = get_unique_key(cur_dict, cur_key)
            cur_value = cur_value[1:-1] if cur_value.startswith('"') and cur_value.endswith('"') else cur_value
            cur_dict[cur_key] = cur_value
        elif "}" in cur_line:
            brackets_depth -= 1
            cur_dict = superior_dicts[brackets_depth]
    return result


def gather_data_with_token_id(data_path):
    token_dirs, gathered_files, cur_token_id = [], {}, 0
    # Detect the deepest dir level where sub dirs are all digits, and regard as tokens level.
    for cur_path, dirs, file_names in os.walk(data_path):
        if len(dirs) == 0:
            continue
        if all([len(ii) < MAX_TOKEN_LEN and str.isdigit(ii) for ii in dirs]):
            dirs = sorted(dirs, key=lambda xx: int(xx))
            token_dirs = [os.path.join(cur_path, dir_name) for dir_name in dirs]  # Search till deepest level
    if len(token_dirs) == 0:
        token_dirs.append(data_path)  # Just use data_path if found no token like dirs

    for token_dir in token_dirs:
        cur_basename = os.path.basename(token_dir)
        cur_token_id = int(cur_basename) if str.isdigit(cur_basename) else 0
        for cur_path, dirs, file_names in os.walk(token_dir):
            file_names = [os.path.join(cur_path, file_name) for file_name in file_names]
            gathered_files.setdefault(cur_token_id, []).extend(file_names)
    return gathered_files


def init_ge_dump_data_from_bin_path(ge_dump_path):
    """
    For data like:
      1/Add.Add_2.44.6.1706596912161941,
      1/Cast.Cast_9.19.6.1706596911887829,
      1/ConcatV2D.ConcatV2.42.6.1706596912161117,

    Return dict:
      {1: {
            'Add_2': '1/Add.Add_2.44.6.1706596912161941',
            'Cast_9': '1/Cast.Cast_9.19.6.1706596911887829',
            'ConcatV2': '1/ConcatV2D.ConcatV2.42.6.1706596912161117',
      }}
    """
    gathered_files = gather_data_with_token_id(ge_dump_path)

    dump_data_with_token_id = {}
    for token_id, file_list in gathered_files.items():
        cur_dump_data = {}
        for file_name in sorted(file_list):
            if os.path.splitext(file_name)[-1] in DUMP_FILE_FILTER_SUFIX:
                continue
            split_name = os.path.basename(file_name).split(".")
            if len(split_name) < 5:
                logger.warning(f"invalid file name: {file_name}, should contain at least 4 '.'")
                continue

            cur_op_name = ".".join(split_name[1:-3])
            if cur_op_name in cur_dump_data:
                exists_file = cur_dump_data[cur_op_name]
                exists_file_size = os.path.getsize(exists_file)
                cur_file_size = os.path.getsize(file_name)
                keep_one = file_name if cur_file_size > exists_file_size else exists_file
                cur_dump_data[cur_op_name] = keep_one
                logger.warning(f"duplicated op name: {cur_op_name}."
                               f" [{os.path.basename(file_name)}, {os.path.basename(exists_file)}]."
                               f" Will keep the larger one {os.path.basename(keep_one)}."
                               )
            else:
                cur_dump_data[cur_op_name] = file_name
        dump_data_with_token_id[token_id] = cur_dump_data
    return dump_data_with_token_id


def init_fx_dump_data_from_path(fx_dump_path):
    """
    For data like:
      1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy,
      1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy,
      1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy,

    Return dict:
      {1: {'mm-aten.mm.default': {
        'input': [
          '1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy',
          '1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy',
        ],
        'output': ['1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy']
      }}}
    """
    gathered_files = gather_data_with_token_id(fx_dump_path)

    dump_data_with_token_id = {}
    for token_id, file_list in gathered_files.items():
        cur_dump_data = {}
        for file_path in sorted(file_list):
            if not file_path.endswith("npy"):
                continue
            file_name = os.path.basename(file_path)
            split_name = file_name.split(".")
            is_input = ".INPUT." in file_name
            cur_op_name = file_name.split('.INPUT.' if is_input else ".OUTPUT.")[0]
            cur_op_map = cur_dump_data.get(cur_op_name, {})
            cur_op_map.setdefault("input" if is_input else "output", []).append(file_path)
            cur_dump_data[cur_op_name] = cur_op_map
        if len(cur_dump_data) > 0:
            dump_data_with_token_id[token_id - 1] = cur_dump_data  # For FX data, token starts from 1, while GE is 0
    return dump_data_with_token_id


def compare_single_data(golden_path, my_path, token_id=0, golden_data=None, my_data=None):
    data_info = BasicDataInfo(golden_path, my_path, token_id)
    return fill_row_data(data_info, loaded_my_data=my_data, loaded_golden_data=golden_data)


""" Comparing GE with FX """


def filter_valid_fx_desc_tensor_info(desc_key, desc_value):
    """Valid one like: 'attr': {'key': '_fx_tensor_name', 'value': {'s': 'add_1-aten.add.Tensor.OUTPUT.0'}}"""
    if not (desc_key == "attr" or desc_key.startswith("attr#")) or not isinstance(desc_value, dict):
        return False
    if desc_value.get("key", None) != "_fx_tensor_name" or not isinstance(desc_value.get("value", None), dict):
        return False
    if not isinstance(desc_value.get("value", {}).get("s", None), str):
        return False
    return True


def get_all_ops_from_fusion_op(op_name, graph_map_dict, ge_dump_data):
    all_ops = []
    while len(op_name) > 0:
        cur_op_name = find_longest_name(op_name, graph_map_dict, ge_dump_data, ge_dump_data)
        if cur_op_name is None or cur_op_name not in graph_map_dict:
            logger.warning(f"Failed parsing ge op name: {cur_op_name}.Compare manually if required.")
            break
        all_ops.append(cur_op_name)
        op_name = op_name[len(cur_op_name):]
    return all_ops


def compare_ge_with_fx(graph_map, ge_dump_data, fx_dump_data, token_id=0):
    gathered_row_data = []
    graph_map_dict = {graph["op"]["name"]: graph["op"] for graph in graph_map if "op" in graph and "name" in graph["op"]}
    for op_name, my_path in ge_dump_data.items():
        all_ops = get_all_ops_from_fusion_op(op_name, graph_map_dict, ge_dump_data)
        if len(all_ops) == 1:
            op_info = graph_map_dict.get(all_ops[0])
            gathered_row_data.extend(compare_ge_with_fx_single_op(op_info, fx_dump_data, op_name, my_path, token_id))
        elif len(all_ops) > 1:
            first_op_info = graph_map_dict.get(all_ops[0])
            last_op_info = graph_map_dict.get(all_ops[-1])
            gathered_row_data.extend(compare_ge_with_fx_multiple_ops(first_op_info, last_op_info, fx_dump_data, op_name,
                                                                     my_path, token_id))
        else:
            op_type = os.path.basename(my_path).split(".")[0]
            if "Cast" in op_type or "TransData" in op_type:
                ge_inputs, ge_outputs = parse_torchair_dump_data(my_path)
                logger.debug(f"ge_inputs length: {len(ge_inputs)}")
                logger.debug(f"ge_outputs length:, {len(ge_outputs)}")
                gathered_row_data.extend(compare_specials_private_ops(ge_inputs, ge_outputs, token_id, my_path))

    return gathered_row_data


def compare_ge_with_fx_single_op(op_info, fx_dump_data, op_name, my_path, token_id=0):
    gathered_row_data = []
    for op_key, op_value in op_info.items():
        if not (op_key == "output_desc" or op_key.startswith("output_desc#")) or not isinstance(op_value, dict):
            continue
        for out_key, out_value in op_value.items():
            if not filter_valid_fx_desc_tensor_info(out_key, out_value):
                continue
            fx_tensor_name = out_value.get("value", {}).get("s", None)
            if fx_tensor_name.split(".")[-2] == "OUTPUT":
                fx_tensor_name = ".".join(fx_tensor_name.split(".")[:-2])
            if fx_tensor_name not in fx_dump_data:
                logger.warning(f"FX data missing, GE tensor name: {op_name}, FX tensor name: {fx_tensor_name}")
                continue

            ge_inputs, ge_outputs = parse_torchair_dump_data(my_path)
            fx_inputs = fx_dump_data.get(fx_tensor_name, {}).get("input", [])
            fx_outputs = fx_dump_data.get(fx_tensor_name, {}).get("output", [])
            logger.debug(f"ge_inputs length: {len(ge_inputs)}, fx_inputs length:, {len(fx_inputs)}")
            logger.debug(f"ge_outputs length: {len(ge_outputs)}, fx_outputs length:, {len(fx_outputs)}")
            gathered_row_data = compare_ops(fx_inputs, fx_outputs, ge_inputs, ge_outputs, token_id, my_path)

    return gathered_row_data


def compare_ge_with_fx_multiple_ops(first_op_info, last_op_info, fx_dump_data, op_name, my_path, token_id):
    gathered_row_data = []
    gathered_row_data.extend(compare_ge_with_fx_multiple_ops_details(first_op_info, fx_dump_data, op_name, my_path
                                                                     , "input", token_id))
    gathered_row_data.extend(compare_ge_with_fx_multiple_ops_details(last_op_info, fx_dump_data, op_name, my_path
                                                                     , "output", token_id))
    return gathered_row_data


def compare_ge_with_fx_multiple_ops_details(op_info: dict, fx_dump_data, op_name, my_path, input_or_output, token_id):
    gathered_row_data = []
    for op_key, op_value in op_info.items():
        if not (op_key == "output_desc" or op_key.startswith("output_desc#")) or not isinstance(op_value, dict):
            continue
        for out_key, out_value in op_value.items():
            if not filter_valid_fx_desc_tensor_info(out_key, out_value):
                continue
            fx_tensor_name = out_value.get("value", {}).get("s", None)
            if fx_tensor_name.split(".")[-2] == "OUTPUT":
                fx_tensor_name = ".".join(fx_tensor_name.split(".")[:-2])
            if fx_tensor_name not in fx_dump_data:
                logger.warning(f"FX data missing, GE tensor name: {op_name}, FX tensor name: {fx_tensor_name}")
                continue
            ge_inputs, ge_outputs = parse_torchair_dump_data(my_path)
            fx_inputs_or_outputs = fx_dump_data.get(fx_tensor_name, {}).get(input_or_output, [])
            ge_input_or_output_path = ""
            ge_inputs_or_outputs = []
            if input_or_output == "input":
                logger.debug(f"ge_inputs length: {len(ge_inputs)}, fx_inputs length:, {len(fx_inputs_or_outputs)}")
                ge_input_or_output_path = "inputs"
                ge_inputs_or_outputs = ge_inputs
            elif input_or_output == "output":
                logger.debug(f"ge_outputs length: {len(ge_outputs)}, fx_outputs length:, {len(fx_inputs_or_outputs)}")
                ge_input_or_output_path = "outputs"
                ge_inputs_or_outputs = ge_outputs
            for cur_id, (fx_input_or_output, ge_input_or_output) in enumerate(zip(fx_inputs_or_outputs,
                                                                                  ge_inputs_or_outputs)):
                cur_ge_data = "{},{},{}".format(my_path, ge_input_or_output_path, cur_id)
                row_data = compare_single_data(fx_input_or_output, cur_ge_data, token_id, my_data=ge_input_or_output)
                gathered_row_data.append(row_data)

    return gathered_row_data


def compare_specials_private_ops(ge_inputs, ge_outputs, token_id, my_path):
    gathered_row_data = []
    for cur_id, (ge_input, ge_output) in enumerate(zip(ge_inputs, ge_outputs)):
        cur_ge_input_data = "{},{},{}".format(my_path, "inputs", cur_id)
        cur_ge_output_data = "{},{},{}".format(my_path, "outputs", cur_id)
        row_data = compare_single_data(cur_ge_input_data, cur_ge_output_data, token_id, ge_input, ge_output)
        gathered_row_data.append(row_data)

    return gathered_row_data


def compare_ops(fx_inputs, fx_outputs, ge_inputs, ge_outputs, token_id, my_path):
    gathered_row_data = []
    for cur_id, (fx_input, ge_input) in enumerate(zip(fx_inputs, ge_inputs)):
        cur_ge_data = "{},{},{}".format(my_path, "inputs", cur_id)
        row_data = compare_single_data(fx_input, cur_ge_data, token_id, my_data=ge_input)
        gathered_row_data.append(row_data)
    for cur_id, (fx_output, ge_output) in enumerate(zip(fx_outputs, ge_outputs)):
        cur_ge_data = "{},{},{}".format(my_path, "outputs", cur_id)
        row_data = compare_single_data(fx_output, cur_ge_data, token_id, my_data=ge_output)
        gathered_row_data.append(row_data)

    return gathered_row_data


""" Comparing fused GE with GE """


def get_all_op_input_names(op_info):
    inputs = [vv for kk, vv in op_info.items() if kk == "input" or kk.startswith("input#")]
    return [":".join(ii.split(":")[:-1]) for ii in inputs]


def find_longest_name(op_name, op_map, fused_ge_dump_data, ge_dump_data):
    if op_name in op_map:
        return op_name
    op_name_len = len(op_name)
    for idx in range(1, op_name_len):
        cur_op_name = op_name[:-idx]
        if cur_op_name in op_map:
            return cur_op_name
        if cur_op_name in fused_ge_dump_data or cur_op_name in ge_dump_data:
            return None  # op_name in dump data but not op_map, abandon
    return None


def gather_fused_op_data(fused_op_name, op_map, fused_ge_dump_data, ge_dump_data):
    gathered_input_names, gathered_inputs, gatherd_input_pathes, gathered_ops = [], [], [], []
    output_path, op_outputs = None, []
    while len(fused_op_name) > 0:
        cur_op_name = find_longest_name(fused_op_name, op_map, fused_ge_dump_data, ge_dump_data)
        if cur_op_name is None or cur_op_name not in op_map:
            logger.warning(f"Failed parsing fused op name: {fused_op_name}. Compare manually if required.")
            break
        cur_input_names = get_all_op_input_names(op_map[cur_op_name])

        if cur_op_name in ge_dump_data:
            cur_path = ge_dump_data[cur_op_name]
            op_inputs, op_outputs = parse_torchair_dump_data(cur_path)
            min_inputs_len = min(len(cur_input_names), len(op_inputs))
            cur_input_names, op_inputs = cur_input_names[:min_inputs_len], op_inputs[:min_inputs_len]
            input_pathes = [",".join([cur_path, "inputs", str(idx)]) for idx in range(min_inputs_len)]
            output_path = cur_path  # Till get the last op path
        else:
            logger.warning(
                f"No dump data for op: {cur_op_name}. Seldom should this happen. Input data matching may be incorrect."
            )
            empty_data = np.array([], dtype='float32')
            op_inputs = [empty_data] * len(cur_input_names)
            input_pathes = [""] * len(cur_input_names)

        gathered_input_names.extend(cur_input_names)
        gathered_ops.append(cur_op_name)
        gathered_inputs.extend(op_inputs)
        gatherd_input_pathes.extend(input_pathes)
        fused_op_name = fused_op_name[len(cur_op_name):]

    filtered_input_names, filtered_inputs, filtered_input_pathes = [], [], []
    for input_name, inputs, input_path in zip(gathered_input_names, gathered_inputs, gatherd_input_pathes):
        if input_name not in gathered_ops:
            filtered_input_names.append(input_name)
            filtered_input_pathes.append(input_path)
            filtered_inputs.append(inputs)
    return (filtered_inputs, filtered_input_pathes), (op_outputs, output_path)  # op_outputs is just the last op output


def compare_ge_with_ge(graph_map, fused_ge_dump_data, ge_dump_data, token_id=0):
    graph_map_dict = {ii["op"]["name"]: ii["op"] for ii in graph_map if "op" in ii and "name" in ii["op"]}
    gathered_row_data = []
    for op_name, my_path in fused_ge_dump_data.items():
        is_fused_op = os.path.basename(my_path).startswith(FUSION_OP_TYPE)
        if is_fused_op:
            (golden_inputs, golden_input_pathes), (golden_outputs, golden_output_path) = gather_fused_op_data(
                op_name, graph_map_dict, fused_ge_dump_data, ge_dump_data
            )
        elif op_name in ge_dump_data:
            golden_path = ge_dump_data[op_name]
            golden_inputs, golden_outputs = parse_torchair_dump_data(golden_path)
            golden_input_pathes = [golden_path] * len(golden_inputs)
            golden_output_path = golden_path
        else:
            logger.warning(f"Golden data missing, My tensor name: {op_name}")
            continue

        my_inputs, my_outputs = parse_torchair_dump_data(my_path)
        logger.debug(f"golden_inputs length: {len(golden_inputs)}, my_inputs length:, {len(my_inputs)}")
        logger.debug(f"golden_outputs length: {len(golden_outputs)}, my_outputs length:, {len(my_outputs)}")

        for cur_id, (golden_input, my_input, golden_input_path) in enumerate(
                zip(golden_inputs, my_inputs, golden_input_pathes)):
            my_path = "{},{},{}".format(my_path, "inputs", cur_id)
            if ",inputs," not in golden_output_path:
                golden_output_path = "{},{},{}".format(golden_output_path, "inputs", cur_id)
            row_data = compare_single_data(
                golden_input_path, my_path, token_id, golden_data=golden_input, my_data=my_input
            )
            gathered_row_data.append(row_data)
        for cur_id, (golden_output, my_output) in enumerate(zip(golden_outputs, my_outputs)):
            my_path = "{},{},{}".format(my_path, "outputs", cur_id)
            golden_output_path = "{},{},{}".format(golden_output_path, "outputs", cur_id)
            row_data = compare_single_data(
                golden_output_path, my_path, token_id, golden_data=golden_output, my_data=my_output
            )
            gathered_row_data.append(row_data)
    return gathered_row_data


""" Main entrance """


def acc_compare(golden_path, my_path, output_path=".", ge_graph_path=None):
    logger.info(f"[compare_torchair], golden_path: {golden_path}, my_path: {my_path}, ge_graph_path: {ge_graph_path}")
    set_msaccucmp_path_from_cann()
    if ge_graph_path is None:
        ge_graph_path = get_torchair_ge_graph_path(my_path)

    graph_map = parse_pbtxt_to_dict(ge_graph_path)
    my_dump_data = init_ge_dump_data_from_bin_path(my_path)
    is_golden_fx = get_torchair_ge_graph_path(golden_path) is None
    if is_golden_fx:
        logger.info("Comparing GE with FX")
        golden_dump_data = init_fx_dump_data_from_path(golden_path)
    else:
        logger.info("Comparing GE with GE")
        golden_dump_data = init_ge_dump_data_from_bin_path(golden_path)

    logger.info(f"All token ids in my_dump_data: {my_dump_data.keys()}")
    logger.info(f"All token ids in golden_dump_data: {golden_dump_data.keys()}")

    gathered_row_data = []
    for token_id in my_dump_data:
        if token_id not in golden_dump_data:
            logger.warning(f"My token_id {token_id} not found in golden dump data")
            continue
        logger.info(f"Comparing token_id: {token_id}")
        if is_golden_fx:
            row_data = compare_ge_with_fx(graph_map, my_dump_data[token_id], golden_dump_data[token_id], token_id)
        else:
            row_data = compare_ge_with_ge(graph_map, my_dump_data[token_id], golden_dump_data[token_id], token_id)
        gathered_row_data.extend(row_data)
    return save_compare_reault_to_csv(gathered_row_data, output_path)
