import sys
import os
import subprocess
import re

import numpy as np

from components.debug.common import logger
from components.debug.common.constant import MSACCUCMP_FILE_PATH


def execute_command(cmd, info_need=True):
    if info_need: 
        logger.info('Execute command:%s' % " ".join(cmd))

    child_process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if child_process.wait() != 0:
        logs = child_process.stdout.read()
        logger.error('Failed to execute command:%s' % " ".join(cmd))
        logger.error(f'\nError log:\n {logs}')
        raise RuntimeError
    

def convert_bin_file_to_npy(bin_file_path, npy_dir_path, cann_path):
    """
    Function Description:
        convert a bin file to npy file.
    Parameter:
        bin_file_path: the path of the bin file needed to be converted to npy
        npy_dir_path: the dest dir to save the converted npy file
        cann_path: user or system cann_path for using msaccucmp.py
    """
    python_version = sys.executable.split('/')[-1]
    msaccucmp_command_file_path = os.path.join(cann_path, MSACCUCMP_FILE_PATH)
    bin2npy_cmd = [python_version, msaccucmp_command_file_path, "convert", "-d", bin_file_path, "-out", npy_dir_path]
    logger.info("convert dump data: %s to npy file" % (bin_file_path))
    execute_command(bin2npy_cmd)


def convert_npy_to_bin(npy_input_path):
    """
    Function Description:
        convert a  file to bin file.
    Parameter:
        npy_file_path: the path of the npy file needed to be converted to bin
    """
    input_initial_path = npy_input_path.split(",")
    outputs = []
    
    for input_item in input_initial_path:
        input_item_path = os.path.realpath(input_item)
        if input_item_path.endswith('.npy'):
            bin_item = input_item[:-4] + '.bin'
            bin_path = input_item_path[:-4] + '.bin'
            npy_data = np.load(input_item_path)

            if os.path.islink(bin_path):
                os.unlink(bin_path)
            if os.path.exists(bin_path):
                os.remove(bin_path)
            npy_data.tofile(bin_path)
            outputs.append(bin_item)
        else:
            outputs.append(input_item)

    return ",".join(outputs)


def parse_input_shape(input_shape):
    input_shapes = {}
    if input_shape == '':
        return input_shapes

    tensor_list = input_shape.split(';')
    for tensor in tensor_list:
        if ':' not in input_shape:
            raise

        tensor_shape_list = tensor.rsplit(':', maxsplit=1)
        if len(tensor_shape_list) == 2:
            shape = tensor_shape_list[1]

            dim_pattern = re.compile(r"^(-?[0-9]{1,100})(,-?[0-9]{1,100}){0,100}")
            match = dim_pattern.match(shape)
            if not match or match.group() is not shape:
                logger.error("www")
                raise RuntimeError
            
            input_shapes[tensor_shape_list[0]] = shape.split(',')    
        else:
            logger.error("wrong parse_input_shape")
            raise RuntimeError
        
    return input_shapes
