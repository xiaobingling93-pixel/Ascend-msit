import argparse
import re
import subprocess
from components.utils.file_open_check import FileStat, is_legal_args_path_string


MAX_SIZE_LIMITE_NORMAL_MODEL = 32 * 1024 * 1024 * 1024  # 32GB


def check_in_path_legality(value):
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"input path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"input path:{path_value} is illegal. Please check.")
    if file_stat.is_file and not file_stat.is_legal_file_type(["onnx"]):
        raise argparse.ArgumentTypeError(f"input path:{path_value} is illegal. Please check.")
    if file_stat.is_file and not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"input path:{path_value} is illegal. Please check.")
    return path_value


def check_in_model_path_legality(value):
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["onnx"]):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_size(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    return path_value


def check_out_model_path_legality(value):
    if not value:
        return None
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('write'):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["onnx"]):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    return path_value


def check_soc(value):
    ivalue = int(value)
    pre_cmd = "npu-smi info -l"
    res = subprocess.run(pre_cmd.split(), shell=False, stdout=subprocess.PIPE)

    tsum = 0
    for line in res.stdout.decode().split('\n'):
        if "Chip Count" in line:
            chip_count = int(line.split()[-1])
            tsum += chip_count
    if ivalue >= tsum or ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value.Please check device id.")
    return ivalue


def check_range(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 64:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value.Range 1 ~ 64.")
    return ivalue


def check_min_num_1(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value.Minimum value 1.")
    return ivalue


def check_min_num_2(value):
    ivalue = int(value)
    if ivalue < -1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value.Minimum value -1.")
    return ivalue


def check_shapes_string(value):
    if not value:
        return value
    shapes_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.-]")
    if regex.search(shapes_string):
        raise argparse.ArgumentTypeError(f"shapes string \"{shapes_string}\" is not a legal string")
    return shapes_string


def check_dtypes_string(value):
    if not value:
        return value
    dtypes_string = value
    regex = re.compile(r"[^_A-Za-z0-9;:/.-]")
    if regex.search(dtypes_string):
        raise argparse.ArgumentTypeError(f"dtypes string \"{dtypes_string}\" is not a legal string")
    return dtypes_string


def check_io_string(value):
    if not value:
        return value
    io_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.-]")
    if regex.search(io_string):
        raise argparse.ArgumentTypeError(f"io string \"{io_string}\" is not a legal string")
    return io_string


def check_nodes_string(value):
    if not value:
        return value
    nodes_string = value
    regex = re.compile(r"[^_A-Za-z0-9,:/.-]")
    if regex.search(nodes_string):
        raise argparse.ArgumentTypeError(f"nodes string \"{nodes_string}\" is not a legal string")
    return nodes_string


def check_single_node_string(value):
    if not value:
        return value
    node_string = value
    regex = re.compile(r"[^_A-Za-z0-9:/.-]")
    if regex.search(node_string):
        raise argparse.ArgumentTypeError(f"single_node string \"{node_string}\" is not a legal string")
    return node_string


def check_normal_string(value):
    if not value:
        return value
    nor_string = value
    regex = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")
    if regex.search(nor_string):
        raise argparse.ArgumentTypeError(f"single_node string \"{nor_string}\" is not a legal string")
    return nor_string


def check_shapes_range_string(value):
    if not value:
        return value
    range_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.\-~]")
    if regex.search(range_string):
        raise argparse.ArgumentTypeError(f"dym range string \"{range_string}\" is not a legal string")
    return range_string


def check_ints_string(value):
    if not value:
        return value
    ints_string = value
    regex = re.compile(r"[^0-9,]")
    if regex.search(ints_string):
        raise argparse.ArgumentTypeError(f"ints string \"{ints_string}\" is not a legal string")
    return ints_string


def check_path_string(value):
    if not value:
        return value
    path_string = value
    if not is_legal_args_path_string(path_string):
        raise argparse.ArgumentTypeError(f"ints string \"{path_string}\" is not a legal string")
    return path_string
