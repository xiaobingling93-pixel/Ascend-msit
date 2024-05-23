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

import logging
import math
import os
import sys
import time
import json
import shutil
import copy
import shlex
import re
import subprocess
import fcntl
from multiprocessing import Pool
from multiprocessing import Manager
import numpy as np

from tqdm import tqdm

from ais_bench.infer.interface import InferSession, MemorySummary
from ais_bench.infer.io_oprations import (create_infileslist_from_inputs_list,
                                          create_pipeline_fileslist_from_inputs_list,
                                          create_intensors_from_infileslist,
                                          get_narray_from_files_list,
                                          get_tensor_from_files_list,
                                          convert_real_files,
                                          PURE_INFER_FAKE_FILE_ZERO,
                                          PURE_INFER_FAKE_FILE_RANDOM,
                                          PURE_INFER_FAKE_FILE, save_tensors_to_file,
                                          get_pure_infer_data)
from ais_bench.infer.summary import summary
from ais_bench.infer.miscellaneous import (dymshape_range_run, get_acl_json_path, version_check,
                                           get_batchsize, ACL_JSON_CMD_LIST)
from ais_bench.infer.utils import (get_file_content, get_file_datasize,
                                   get_fileslist_from_dir, list_split, list_share,
                                   save_data_to_files, create_fake_file_name, logger,
                                   create_tmp_acl_json, move_subdir, convert_helper)
from ais_bench.infer.path_security_check import is_legal_args_path_string
from ais_bench.infer.args_adapter import BenchMarkArgsAdapter
from ais_bench.infer.backends import BackendFactory
from ais_bench.infer.path_security_check import ms_open, MAX_SIZE_LIMITE_CONFIG_FILE

PERMISSION_DIR = 0o750
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def set_session_options(session, args):
    # 增加校验
    aipp_batchsize = -1
    if args.dym_batch != 0:
        session.set_dynamic_batchsize(args.dym_batch)
        aipp_batchsize = session.get_max_dym_batchsize()
    elif args.dym_hw is not None:
        hwstr = args.dym_hw.split(",")
        session.set_dynamic_hw((int)(hwstr[0]), (int)(hwstr[1]))
    elif args.dym_dims is not None:
        session.set_dynamic_dims(args.dym_dims)
    elif args.dym_shape is not None:
        session.set_dynamic_shape(args.dym_shape)
    else:
        session.set_staticbatch()

    if args.batchsize is None:
        args.batchsize = get_batchsize(session, args)
        logger.info(f"try get model batchsize:{args.batchsize}")

    if not args.auto_set_dymshape_mode and not args.auto_set_dymdims_mode:
        if args.batchsize < 0 and not args.dym_batch and not args.dym_dims and not args.dym_shape:
            raise RuntimeError('dynamic batch om model detected, but dymbatch, dymdims or dymshape not set!')

    if aipp_batchsize < 0:
        aipp_batchsize = args.batchsize

    # 确认模型只有一个动态 aipp input
    if args.dym_shape is not None or args.auto_set_dymshape_mode:
        aipp_input_exist = 0
    else:
        aipp_input_exist = session.get_dym_aipp_input_exist()
    logger.debug(f"aipp_input_exist: {aipp_input_exist}")
    if (args.aipp_config is not None) and (aipp_input_exist == 1):
        session.load_aipp_config_file(args.aipp_config, aipp_batchsize)
        session.check_dym_aipp_input_exist()
    elif (args.aipp_config is None) and (aipp_input_exist == 1):
        logger.error("can't find aipp config file for model with dym aipp input , please check it!")
        raise RuntimeError('aipp model without aipp config!')
    elif (aipp_input_exist > 1):
        logger.error(f"don't support more than one dynamic aipp input in model, \
                     amount of aipp input is {aipp_input_exist}")
        raise RuntimeError('aipp model has more than 1 aipp input!')
    elif (aipp_input_exist == -1):
        raise RuntimeError('aclmdlGetAippType failed!')

    # 设置custom out tensors size
    if args.output_size is not None:
        customsizes = [int(n) for n in args.output_size.split(',')]
        logger.debug(f"set customsize:{customsizes}")
        session.set_custom_outsize(customsizes)


def init_inference_session(args, acl_json_path):
    session = InferSession(args.device, args.model, acl_json_path, args.debug, args.loop)

    set_session_options(session, args)
    logger.debug(f"session info:{session.session}")
    return session


def set_dymshape_shape(session, inputs):
    shape_list = []
    intensors_desc = session.get_inputs()
    for i, input_ in enumerate(inputs):
        str_shape = [str(shape) for shape in input_.shape]
        shapes = ",".join(str_shape)
        dyshape = f"{intensors_desc[i].name}:{shapes}"
        shape_list.append(dyshape)
    dyshapes = ';'.join(shape_list)
    logger.debug(f"set dymshape shape:{dyshapes}")
    session.set_dynamic_shape(dyshapes)
    summary.add_batchsize(inputs[0].shape[0])


def set_dymdims_shape(session, inputs):
    shape_list = []
    intensors_desc = session.get_inputs()
    for i, input_ in enumerate(inputs):
        str_shape = [str(shape) for shape in input_.shape]
        shapes = ",".join(str_shape)
        dydim = f"{intensors_desc[i].name}:{shapes}"
        shape_list.append(dydim)
    dydims = ';'.join(shape_list)
    logger.debug(f"set dymdims shape:{dydims}")
    session.set_dynamic_dims(dydims)
    summary.add_batchsize(inputs[0].shape[0])


def warmup(session, args, intensors_desc, infiles):
    # prepare input data
    infeeds = []
    for j, files in enumerate(infiles):
        if args.run_mode == "tensor":
            tensor = get_tensor_from_files_list(files, session, intensors_desc[j].realsize,
                                                args.pure_data_type, args.no_combine_tensor_mode)
            infeeds.append(tensor)
        else:
            narray = get_narray_from_files_list(files, intensors_desc[j].realsize,
                                                args.pure_data_type, args.no_combine_tensor_mode)
            infeeds.append(narray)
    session.set_loop_count(1)
    # warmup
    for _ in range(args.warmup_count):
        outputs = run_inference(session, args, infeeds, out_array=True)

    session.set_loop_count(args.loop)

    # reset summary info
    summary.reset()
    session.reset_summaryinfo()
    MemorySummary.reset()
    logger.info(f"warm up {args.warmup_count} done")


def run_inference(session, args, inputs, out_array=False):
    if args.auto_set_dymshape_mode:
        set_dymshape_shape(session, inputs)
    elif args.auto_set_dymdims_mode:
        set_dymdims_shape(session, inputs)
    outputs = session.run(inputs, out_array)
    return outputs


def run_pipeline_inference(session, args, infileslist, output_prefix, extra_session):
    out = output_prefix if output_prefix is not None else ""
    pure_infer_mode = False
    if args.input is None:
        pure_infer_mode = True
    session.run_pipeline(infileslist,
                         out,
                         args.auto_set_dymshape_mode,
                         args.auto_set_dymdims_mode,
                         args.outfmt,
                         pure_infer_mode,
                         [s.session for s in extra_session])


# tensor to loop infer
def infer_loop_tensor_run(session, args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference tensor Processing')):
        intensors = []
        for j, files in enumerate(infiles):
            tensor = get_tensor_from_files_list(files, session, intensors_desc[j].realsize,
                                                args.pure_data_type, args.no_combine_tensor_mode)
            intensors.append(tensor)
        outputs = run_inference(session, args, intensors)
        session.convert_tensors_to_host(outputs)
        if output_prefix is not None:
            save_tensors_to_file(
                outputs, output_prefix, infiles,
                args.outfmt, i, args.output_batchsize_axis
            )


# files to loop iner
def infer_loop_files_run(session, args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference files Processing')):
        intensors = []
        for j, files in enumerate(infiles):
            real_files = convert_real_files(files)
            tensor = session.create_tensor_from_fileslist(intensors_desc[j], real_files)
            intensors.append(tensor)
        outputs = run_inference(session, args, intensors)
        session.convert_tensors_to_host(outputs)
        if output_prefix is not None:
            save_tensors_to_file(
                outputs, output_prefix, infiles,
                args.outfmt, i, args.output_batchsize_axis
            )


# First prepare the data, then execute the reference, and then write the file uniformly
def infer_fulltensors_run(session, args, intensors_desc, infileslist, output_prefix):
    outtensors = []
    intensorslist = create_intensors_from_infileslist(infileslist, intensors_desc, session,
                                                      args.pure_data_type, args.no_combine_tensor_mode)

    for inputs in tqdm(intensorslist, file=sys.stdout, desc='Inference Processing full'):
        outputs = run_inference(session, args, inputs)
        outtensors.append(outputs)

    for i, outputs in enumerate(outtensors):
        session.convert_tensors_to_host(outputs)
        if output_prefix is not None:
            save_tensors_to_file(
                outputs, output_prefix, infileslist[i],
                args.outfmt, i, args.output_batchsize_axis
            )


# loop numpy array to infer
def infer_loop_array_run(session, args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference array Processing')):
        innarrays = []
        for j, files in enumerate(infiles):
            narray = get_narray_from_files_list(files, intensors_desc[j].realsize, args.pure_data_type)
            innarrays.append(narray)
        outputs = run_inference(session, args, innarrays)
        session.convert_tensors_to_host(outputs)
        if args.output is not None:
            save_tensors_to_file(
                outputs, output_prefix, infiles,
                args.outfmt, i, args.output_batchsize_axis
            )


def infer_pipeline_run(session, args, infileslist, output_prefix, extra_session):
    logger.info(f"run in pipeline mode with computing threadsnumber:{args.threads}")
    run_pipeline_inference(session, args, infileslist, output_prefix, extra_session)


def get_file_name(file_path: str, suffix: str, res_file_path: list) -> list:
    """获取路径下的指定文件类型后缀的文件
    Args:
        file_path: 文件夹的路径
        suffix: 要提取的文件类型的后缀
        res_file_path: 保存返回结果的列表
    Returns: 文件路径
    """
    for file in os.listdir(file_path):

        if os.path.isdir(os.path.join(file_path, file)):
            get_file_name(os.path.join(file_path, file), suffix, res_file_path)
        else:
            res_file_path.append(os.path.join(file_path, file))
    # endswith：表示以suffix结尾。可根据需要自行修改；如：startswith：表示以suffix开头，__contains__：包含suffix字符串
    if suffix == '' or suffix is None:
        return res_file_path
    else:
        return list(filter(lambda x: x.endswith(suffix), res_file_path))


def get_legal_json_content(acl_json_path):
    cmd_dict = {}
    with ms_open(acl_json_path, mode="r", max_size=MAX_SIZE_LIMITE_CONFIG_FILE) as f:
        json_dict = json.load(f)
    profile_dict = json_dict.get("profiler")
    for option_cmd in ACL_JSON_CMD_LIST:
        if profile_dict.get(option_cmd):
            if option_cmd == "output" and not is_legal_args_path_string(profile_dict.get(option_cmd)):
                raise Exception(f"output path in acl_json is illegal!")
            cmd_dict.update({"--" + option_cmd.replace('_', '-'): profile_dict.get(option_cmd)})
            if (option_cmd == "sys_hardware_mem_freq"):
                cmd_dict.update({"--sys-hardware-mem": "on"})
            if (option_cmd == "sys_interconnection_freq"):
                cmd_dict.update({"--sys-interconnection-profiling": "on"})
            if (option_cmd == "dvpp_freq"):
                cmd_dict.update({"--dvpp-profiling": "on"})
    return cmd_dict


def json_to_msprof_cmd(acl_json_path):
    json_dict = get_legal_json_content(acl_json_path)
    msprof_option_cmd = " ".join([f"{key}={value}" for key, value in json_dict.items()])
    return msprof_option_cmd


def regenerate_cmd(args:BenchMarkArgsAdapter):
    args_dict = args.get_all_args_dict()
    cmd = sys.executable + " -m ais_bench"
    for key, value in args_dict.items():
        if key == '--acl_json_path':
            continue
        if key == '--warmup_count':
            cmd = cmd + " " + f"{key}={0}"
            continue
        if key == '--profiler':
            cmd = cmd + " " + f"{key}={0}"
            continue
        if value:
            cmd = cmd + " " + f"{key}={value}"
    return cmd


def msprof_run_profiling(args, msprof_bin):
    if args.acl_json_path is not None:
        # acl.json to msprof cmd
        args.profiler_rename = False
        cmd = regenerate_cmd(args)
        msprof_cmd = f"{msprof_bin} --application=\"{cmd}\" " + json_to_msprof_cmd(args.acl_json_path)
    else:
        # default msprof cmd
        cmd = regenerate_cmd(args)
        msprof_cmd = f"{msprof_bin} --output={args.output}/profiler --application=\"{cmd}\" --model-execution=on \
                    --sys-hardware-mem=on --sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off \
                    --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on" \

    ret = -1
    msprof_cmd_list = shlex.split(msprof_cmd)
    logger.info(f"msprof cmd:{msprof_cmd} begin run")
    if (args.profiler_rename):
        p = subprocess.Popen(msprof_cmd_list, stdout=subprocess.PIPE, shell=False, bufsize=0)
        flags = fcntl.fcntl(p.stdout, fcntl.F_GETFL)
        fcntl.fcntl(p.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        get_path_flag = True
        sub_str = ""
        for line in iter(p.stdout.read, b''):
            if not line:
                continue
            line = line.decode()
            if (get_path_flag and line.find("PROF_") != -1):
                get_path_flag = False
                start_index = line.find("PROF_")
                sub_str = line[start_index:(start_index + 46)] # PROF_XXXX的目录长度为46
            print(f'{line}', flush=True, end="")
        p.stdout.close()
        p.wait()

        output_prefix = os.path.join(args.output, "profiler")
        output_prefix = os.path.join(output_prefix, sub_str)
        hash_str = sub_str.rsplit('_')[-1]
        file_name = get_file_name(output_prefix, ".csv", [])
        file_name_json = get_file_name(output_prefix, ".json", [])

        model_name = os.path.basename(args.model).split(".")[0]
        for file in file_name:
            real_file = os.path.splitext(file)[0]
            os.rename(file, real_file + "_" + model_name + "_" + hash_str + ".csv")
        for file in file_name_json:
            real_file = os.path.splitext(file)[0]
            os.rename(file, real_file + "_" + model_name + "_" + hash_str + ".json")
        ret = 0
    else:
        ret = subprocess.call(msprof_cmd_list, shell=False)
        logger.info(f"msprof cmd:{msprof_cmd} end run ret:{ret}")
    return ret


def get_energy_consumption(npu_id):
    cmd = f"npu-smi info -t power -i {npu_id}"
    get_npu_id = subprocess.run(cmd.split(), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if get_npu_id.returncode != 0:
        raise RuntimeError(f"Invalid npu id:{npu_id}, exec cmd: 'npu-smi info' to check valid npu_id")
    npu_id = get_npu_id.stdout.decode('gb2312')
    power = []
    npu_id = npu_id.split("\n")
    for key in npu_id:
        if key.find("Power Dissipation(W)", 0, len(key)) != -1:
            power = key[34:len(key)]
            break

    return power


def convert(tmp_acl_json_path, real_dump_path, tmp_dump_path):
    if real_dump_path is not None and tmp_dump_path is not None:
        output_dir, timestamp = move_subdir(tmp_dump_path, real_dump_path)
        convert_helper(output_dir, timestamp)
    if tmp_dump_path is not None:
        shutil.rmtree(tmp_dump_path)
    if tmp_acl_json_path is not None:
        os.remove(tmp_acl_json_path)


def main(args, index=0, msgq=None, device_list=None):
    # if msgq is not None,as subproces run
    if msgq is not None:
        logger.info(f"subprocess_{index} main run")

    if args.debug:
        logger.setLevel(logging.DEBUG)

    acl_json_path = get_acl_json_path(args)
    tmp_acl_json_path = None
    if args.dump_npy and acl_json_path is not None:
        tmp_acl_json_path, real_dump_path, tmp_dump_path = create_tmp_acl_json(acl_json_path)

    session = init_inference_session(args, tmp_acl_json_path if tmp_acl_json_path is not None else acl_json_path)
    # if pipeline is set and threads number is > 1, create a session pool for extra computing
    extra_session = []
    if args.pipeline:
        extra_session = [init_inference_session(args, tmp_acl_json_path if tmp_acl_json_path is not None\
                                                else acl_json_path) for _ in range(args.threads - 1)]

    intensors_desc = session.get_inputs()
    if device_list is not None and len(device_list) > 1:
        if args.output is not None:
            if args.output_dirname is None:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                output_prefix = os.path.join(args.output, timestr)
                output_prefix = os.path.join(output_prefix, "device" + str(device_list[index]) + "_" + str(index))
            else:
                output_prefix = os.path.join(args.output, args.output_dirname)
                output_prefix = os.path.join(output_prefix, "device" + str(device_list[index]) + "_" + str(index))
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix, PERMISSION_DIR)
            os.chmod(args.output, PERMISSION_DIR)
            logger.info(f"output path:{output_prefix}")
        else:
            output_prefix = None
    else:
        if args.output is not None:
            if args.output_dirname is None:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                output_prefix = os.path.join(args.output, timestr)
            else:
                output_prefix = os.path.join(args.output, args.output_dirname)
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix, PERMISSION_DIR)
            os.chmod(args.output, PERMISSION_DIR)
            logger.info(f"output path:{output_prefix}")
        else:
            output_prefix = None

    inputs_list = [] if args.input is None else args.input.split(',')

    # create infiles list accord inputs list
    if len(inputs_list) == 0:
        # Pure reference scenario. Create input zero data
        if not args.pipeline:
            infileslist = [[[PURE_INFER_FAKE_FILE] for _ in intensors_desc]]
        else:
            infileslist = [[]]
            pure_file = PURE_INFER_FAKE_FILE_ZERO if args.pure_data_type == "zero" else PURE_INFER_FAKE_FILE_RANDOM
            for _ in intensors_desc:
                infileslist[0].append(pure_file)
    else:
        if not args.pipeline:
            infileslist = create_infileslist_from_inputs_list(inputs_list, intensors_desc, args.no_combine_tensor_mode)
        else:
            infileslist = create_pipeline_fileslist_from_inputs_list(inputs_list, intensors_desc)
    if not args.pipeline:
        warmup(session, args, intensors_desc, infileslist[0])
    else:
        # prepare for pipeline case
        infiles = []
        for file in infileslist[0]:
            infiles.append([file])
        warmup(session, args, intensors_desc, infiles)
        for sess in extra_session:
            warmup(sess, args, intensors_desc, infiles)

    if args.pipeline and (args.auto_set_dymshape_mode or args.auto_set_dymdims_mode):
        for file_list in infileslist:
            input_first = np.load(file_list[0])
            summary.add_batchsize(input_first.shape[0])

    if msgq is not None:
        # wait subprocess init ready, if time eplapsed, force ready run
        logger.info(f"subprocess_{index} qsize:{msgq.qsize()} now waiting")
        msgq.put(index)
        time_sec = 0
        while True:
            if msgq.qsize() >= args.subprocess_count:
                break
            time_sec = time_sec + 1
            if time_sec > 10:
                logger.warning(f"subprocess_{index} qsize:{msgq.qsize()} time:{time_sec} s elapsed")
                break
            time.sleep(1)
        logger.info(f"subprocess_{index} qsize:{msgq.qsize()} ready to infer run")

    start_time = time.time()
    if args.energy_consumption:
        start_energy_consumption = get_energy_consumption(args.npu_id)
    if args.pipeline:
        infer_pipeline_run(session, args, infileslist, output_prefix, extra_session)
    else:
        run_mode_switch = {
            "array": infer_loop_array_run,
            "files": infer_loop_files_run,
            "full": infer_fulltensors_run,
            "tensor": infer_loop_tensor_run
        }
        if run_mode_switch.get(args.run_mode) is not None:
            run_mode_switch.get(args.run_mode)(session, args, intensors_desc, infileslist, output_prefix)
        else:
            raise RuntimeError(f'wrong run_mode:{args.run_mode}')
    if args.energy_consumption:
        end_energy_consumption = get_energy_consumption(args.npu_id)
    end_time = time.time()

    multi_threads_mode = args.threads > 1 and args.pipeline
    summary.add_args(sys.argv)
    s = session.summary()
    if multi_threads_mode:
        summary.npu_compute_time_interval_list = s.exec_time_list
    else:
        summary.npu_compute_time_list = [end_time - start_time for start_time, end_time in s.exec_time_list]
    summary.h2d_latency_list = MemorySummary.get_h2d_time_list()
    summary.d2h_latency_list = MemorySummary.get_d2h_time_list()
    summary.report(args.batchsize, output_prefix, args.display_all_summary, multi_threads_mode)
    try:
        if args.energy_consumption:
            energy_consumption = ((float(end_energy_consumption) + float(start_energy_consumption)) / 2.0) \
                * (end_time - start_time)
            logger.info(f"NPU ID:{args.npu_id} energy consumption(J):{energy_consumption}")
    except AttributeError as err:
        logger.error(f"Attribute Access Error: {err}")
        raise RuntimeError("Error accessing an attribute, please verify if the NPU ID is correct. ") from err
    except Exception as err:
        logger.error(f"Unexpected Error: {err}")
        raise RuntimeError(
            "Energy consumption append an unexpected error occurred, please check the input parameters.") from err

    if msgq is not None:
        # put result to msgq
        msgq.put([index, summary.infodict['throughput'], start_time, end_time])

    session.free_resource()
    for sess in extra_session:
        sess.free_resource()

    InferSession.finalize()

    if args.dump_npy and acl_json_path is not None:
        convert(tmp_acl_json_path, real_dump_path, tmp_dump_path)


def print_subproces_run_error(value):
    logger.error(f"subprocess run failed error_callback:{value}")


def seg_input_data_for_multi_process(args, inputs, jobs):
    inputs_list = [] if inputs is None else inputs.split(',')
    if inputs_list is None:
        return inputs_list

    fileslist = []
    if os.path.isfile(inputs_list[0]):
        fileslist = inputs_list
    elif os.path.isdir(inputs_list[0]):
        for dir_path in inputs_list:
            fileslist.extend(get_fileslist_from_dir(dir_path))
    else:
        logger.error(f'error {inputs_list[0]} not file or dir')
        raise RuntimeError()

    args.device = 0
    acl_json_path = get_acl_json_path(args)
    session = init_inference_session(args, acl_json_path)
    intensors_desc = session.get_inputs()
    try:
        chunks_elements = math.ceil(len(fileslist) / len(intensors_desc))
    except ZeroDivisionError as err:
        logger.error("ZeroDivisionError: intensors_desc is empty")
        raise RuntimeError("error zero division") from err
    chunks = list(list_split(fileslist, chunks_elements, None))
    fileslist = [[] for _ in range(jobs)]
    for _, chunk in enumerate(chunks):
        try:
            splits_elements = int(len(chunk) / jobs)
        except ZeroDivisionError as err:
            logger.error("ZeroDivisionError: intensors_desc is empty")
            raise RuntimeError("error zero division") from err
        splits_left = len(chunk) % jobs
        splits = list(list_share(chunk, jobs, splits_elements, splits_left))
        for j, split in enumerate(splits):
            fileslist[j].extend(split)
    res = []
    for files in fileslist:
        res.append(','.join(list(filter(None, files))))
    return res


def multidevice_run(args):
    logger.info(f"multidevice:{args.device} run begin")
    device_list = args.device
    npu_id_list = args.npu_id
    p = Pool(len(device_list))
    msgq = Manager().Queue()
    args.subprocess_count = len(device_list)
    splits = None
    if (args.input is not None and args.divide_input):
        jobs = args.subprocess_count
        splits = seg_input_data_for_multi_process(args, args.input, jobs)

    for i, device in enumerate(device_list):
        cur_args = copy.deepcopy(args)
        cur_args.device = int(device)
        if args.energy_consumption:
            cur_args.npu_id = int(npu_id_list[i])
        if args.divide_input:
            cur_args.input = None if splits is None else list(splits)[i]
        p.apply_async(main, args=(cur_args, i, msgq, device_list), error_callback=print_subproces_run_error)

    p.close()
    p.join()
    result = 0 if 2 * len(device_list) == msgq.qsize() else 1
    logger.info(f"multidevice run end qsize:{msgq.qsize()} result:{result}")
    tlist = []
    while msgq.qsize() != 0:
        ret = msgq.get()
        if type(ret) == list:
            logger.info(f"i:{ret[0]} device_{device_list[ret[0]]} throughput:{ret[1]} \
                start_time:{ret[2]} end_time:{ret[3]}")
            tlist.append(ret[1])
    logger.info(f'summary throughput:{sum(tlist)}')
    return result


def args_rules(args):
    if args.profiler and args.dump:
        logger.error("parameter --profiler cannot be true at the same time as parameter --dump, please check them!\n")
        raise RuntimeError('error bad parameters --profiler and --dump')

    if args.output_dirname and args.output_dirname[0] == '/': # abspath is not permitted
        raise ValueError("--output_dirname do not support abs path!" )

    if (args.profiler or args.dump) and (args.output is None):
        logger.error("when dump or profiler, miss output path, please check them!")
        raise RuntimeError('miss output parameter!')

    if not args.auto_set_dymshape_mode and not args.auto_set_dymdims_mode:
        args.no_combine_tensor_mode = False
    else:
        args.no_combine_tensor_mode = True

    if args.profiler and args.warmup_count != 0 and args.input is not None:
        logger.info("profiler mode with input change warmup_count to 0")
        args.warmup_count = 0

    if args.output is None and args.output_dirname is not None:
        logger.error(
            "parameter --output_dirname cann't be used alone. Please use it together with the parameter --output!\n")
        raise RuntimeError('error bad parameters --output_dirname')

    if args.threads > 1 and not args.pipeline:
        logger.info("need to set --pipeline when setting threads number to be more than one.")
        args.threads = 1

    return args


def acl_json_base_check(args):
    if args.acl_json_path is None:
        return args
    json_path = args.acl_json_path
    try:
        with ms_open(json_path, mode="r", max_size=MAX_SIZE_LIMITE_CONFIG_FILE) as f:
            json_dict = json.load(f)
    except Exception as err:
        logger.error(f"can't read acl_json_path:{json_path}")
        raise Exception from err
    if json_dict.get("profiler") is not None and json_dict.get("profiler").get("switch") == "on":
        args.profiler = True
    if json_dict.get("dump") is not None:
        args.profiler = False
    return args


def config_check(config_path):
    if not config_path:
        return
    max_config_size = 12800
    if os.path.splitext(config_path)[1] != ".config":
        logger.error(f"aipp_config:{config_path} is not a .config file")
        raise TypeError(f"aipp_config:{config_path} is not a .config file")
    config_size = os.path.getsize(config_path)
    if config_size > max_config_size:
        logger.error(f"aipp_config_size:{config_size} byte out of max limit {max_config_size} byte")
        raise MemoryError(f"aipp_config_size:{config_size} byte out of max limit")
    return


def backend_run(args):
    backend_class = BackendFactory.create_backend(args.backend)
    backend = backend_class(args)
    backend.load(args.model)
    backend.run()
    perf = backend.get_perf()
    logger.info(f"perf info:{perf}")


def benchmark_process(args:BenchMarkArgsAdapter):
    args = args_rules(args)
    version_check(args)
    args = acl_json_base_check(args)

    if args.perf:
        backend_run(args)
        return 0

    if args.profiler:
        # try use msprof to run
        msprof_bin = shutil.which('msprof')
        if msprof_bin is None:
            logger.info("find no msprof continue use acl.json mode, result won't be parsed as csv")
        elif os.getenv('AIT_NO_MSPROF_MODE') == '1':
            logger.info("find AIT_NO_MSPROF_MODE set, continue use acl.json mode, result won't be parsed as csv")
        else:
            ret = msprof_run_profiling(args, msprof_bin)
            return ret

    if args.dym_shape_range is not None and args.dym_shape is None:
        # dymshape range run,according range to run each shape infer get best shape
        dymshape_range_run(args)
        return 0

    if type(args.device) == list:
        # args has multiple device, run single process for each device
        ret = multidevice_run(args)
        return ret

    main(args)
    return 0
