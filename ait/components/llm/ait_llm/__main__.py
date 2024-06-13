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
import subprocess

from components.utils.parser import BaseCommand
from ait_llm.dump.initial import init_dump_task, clear_dump_task
from ait_llm.opcheck.opchecker import OpChecker, NAMEDTUPLE_PRECISION_METRIC, NAMEDTUPLE_PRECISION_MODE
from ait_llm.errcheck.process import process_error_check
from ait_llm.common.utils import str2bool, check_positive_integer, check_device_integer, safe_string, check_exec_cmd, \
    check_ids_string, check_number_list, check_output_path_legality, check_input_path_legality
from ait_llm.common.log import logger, set_log_level, LOG_LEVELS


LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


class DumpCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--only-save-desc',
            '-sd',
            required=False,
            dest="save_desc",
            action='store_true',
            default=False,
            help='0 When save tensor, 1 When only save tensor description instead of tensor')

        parser.add_argument(
            '--save-operation-ids',
            '-ids',
            required=False,
            dest="ids",
            type=check_ids_string,
            default="",
            help='Save Tensor Ids')

        parser.add_argument(
            '--execute-range',
            '-er',
            required=False,
            dest="range",
            type=check_number_list,
            default="0,0",
            help='The range of saving tensor.Eg:0,10')

        parser.add_argument(
            '--save-operation-child',
            '-child',
            required=False,
            dest="child",
            type=str2bool,
            default=True,
            help='Dump all data of child operations if True, do nothing if False.')

        parser.add_argument(
            '--save-time',
            '-time',
            required=False,
            dest="time",
            type=check_positive_integer,
            default=1,
            help='0 when only need dump data before execution, '
                 '1 when only need dump data after execution, 2 both.')

        parser.add_argument(
            '--operation-name',
            '-opname',
            required=False,
            dest="opname",
            type=safe_string,
            default=None,
            help='Operation names need to dump.')

        parser.add_argument(
            '--save-tiling',
            '-tiling',
            required=False,
            dest="tiling",
            action='store_true',
            default=False,
            help='Dump all data of child operations if True, do nothing if False')

        parser.add_argument(
            '--exec',
            dest="exec",
            required=True,
            type=safe_string,
            default='',
            help='Exec command to run acltransformer model inference.'
                 'E.g: --exec \"bash run.sh patches/models/modeling_xxx.py\" ')

        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            type=check_output_path_legality,
            default='./',
            help='Data output directory.E.g:--output /xx/xxxx/xx')

        parser.add_argument(
            '--save-tensor-part',
            '-stp',
            required=False,
            dest="save_tensor_part",
            type=check_positive_integer,
            default=2,
            help='0 when only need dump intensor, '
                 '1 when only need dump outtensor, 2 both.')

        parser.add_argument(
            '--type',
            dest="type",
            required=False,
            nargs='+',
            default=['tensor'],
            choices=['model', 'layer', 'op', 'kernel', 'tensor', 'cpu_profiling', 'onnx'],
            help='dump type.')

        parser.add_argument(
            '--device-id',
            '-device',
            required=False,
            dest="device_id",
            type=check_positive_integer,
            default=None,
            help='Specify a single device ID for dumping data, will skip other devices.')

        parser.add_argument("--log-level", "-l", default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")

    def handle(self, args, **kwargs):
        if args.exec:
            set_log_level(args.log_level)
            logger.info(f"About to execute command : {args.exec}")
            logger.warning("Please ensure that your execution command is secure.")
            init_dump_task(args)
            # 有的大模型推理任务启动后，输入对话时有提示符，使用subprocess拉起子进程无法显示提示符
            cmds = args.exec.split()
            subprocess.run(cmds, shell=False)
            clear_dump_task(args)
            return


class CompareCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs):
        parser.add_argument(
            '--golden-path',
            '-gp',
            dest="golden_path",
            required=True,
            type=check_input_path_legality,
            help='Golden data path. It supports directory or file.')

        parser.add_argument(
            '--my-path',
            '-mp',
            dest="my_path",
            required=True,
            type=check_input_path_legality,
            help='Compared data path. It supports directory or file.')

        parser.add_argument(
            '--cmp-level',
            '-cl',
            dest="cmp_level",
            required=False,
            default="layer",
            choices=["layer", "token"],
            help='Compare level. only enabled for atb.')

        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            type=check_output_path_legality,
            default='./',
            help='Data output directory.E.g:--output /xx/xxxx/xx')

        parser.add_argument(
            '--op-mapping-file',
            '-mf',
            dest="mapping_file",
            required=False,
            type=check_output_path_legality,
            default='',
            help='Operation mapping file directory.E.g:--op-mapping-file /xx/xxxx/xx')

        parser.add_argument(
            '--custom-algorithms',
            '-alg',
            required=False,
            nargs='+',
            help='custom comparing algorithms in format "python_file_path.py:function". \
                  Should better be a standalong file, and function should in format like \
                  "def foo(golden_tensor, my_tensor): return float_value, string_message"')

        parser.add_argument("--log-level", "-l", default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")

    def handle(self, args, **kwargs):
        from ait_llm.compare.torchair_acc_cmp import get_torchair_ge_graph_path

        set_log_level(args.log_level)

        # Adding custom comparing algorithms
        if args.custom_algorithms:
            from ait_llm.compare.cmp_algorithm import register_custom_compare_algorithm

            for custom_compare_algorithm in args.custom_algorithms:
                register_custom_compare_algorithm(custom_compare_algorithm)

        # accuracy comparing for different scenarios
        torchair_ge_graph_path = get_torchair_ge_graph_path(args.my_path)
        if torchair_ge_graph_path is not None:
            from ait_llm.compare.torchair_acc_cmp import acc_compare

            acc_compare(args.golden_path, args.my_path, args.output, torchair_ge_graph_path)
        else:
            from ait_llm.compare.atb_acc_cmp import acc_compare
            from ait_llm.compare.cmp_mgr import CompareMgr
            comared = acc_compare(os.path.abspath(args.golden_path), os.path.abspath(args.my_path),
                        args.output, args.mapping_file, args.cmp_level)
            if not comared:
                cmpMgr = CompareMgr(os.path.abspath(args.golden_path), os.path.abspath(args.my_path), args)
                if cmpMgr.is_parsed_cmp_path():
                    cmpMgr.compare(args.output)


class OpcheckCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs):
        parser.add_argument(
            '--input',
            '-i',
            required=True,
            type=check_input_path_legality,
            help='input directory.E.g:--input OUTPUT_DIR/PID_TID/0/')

        parser.add_argument(
            '--output',
            '-o',
            required=False,
            type=check_output_path_legality,
            default='./',
            help='Data output directory.E.g:--output /xx/xxxx/xx')

        parser.add_argument(
            '--operation-ids',
            '-ids',
            required=False,
            type=check_ids_string,
            default="",
            help='Save Tensor Ids.E.g:-ids 24_1,2_3_5')

        parser.add_argument(
            '--operation-name',
            '-opname',
            required=False,
            type=safe_string,
            default=None,
            help='Operation names need to dump.E.g:-opname self,linear')

        parser.add_argument(
            '--precision-metric',
            '-metric',
            required=False,
            nargs='+',
            default=[],
            choices=NAMEDTUPLE_PRECISION_METRIC._fields,
            help='Output more results of other precision metrics.E.g:-metric abs kl cos_sim')

        parser.add_argument(
            '--device-id',
            '-device',
            required=False,
            type=check_device_integer,
            default=0,
            help='Spicifies the NPU device to bu used.E.g.:-device 1')

        parser.add_argument(
            '--atb-rerun',
            '-rerun',
            required=False,
            action='store_true',
            default=False,
            help='Rerun atb operations if True. Compare outputs in dump data if False')

        parser.add_argument(
            '--custom-algorithms',
            '-alg',
            required=False,
            nargs='+',
            help='custom comparing algorithms in format "python_file_path.py:function". \
                  Should better be a standalong file, and function should in format like \
                  "def foo(golden_tensor, my_tensor): return float_value, string_message"')

        parser.add_argument(
            '--precision-mode',
            '-pmode',
            required=False,
            default="keep_origin_dtype",
            choices=NAMEDTUPLE_PRECISION_MODE._fields,
            help='Specifies the precision mode to calculate golden output. Keep origin dtype or translate all \
                float tensors to torch.float16/torch.float32 before calculating and comparing.E.g.:-pmode force_fp32')

        parser.add_argument("-l", "--log-level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs):
        set_log_level(args.log_level)

        # Adding custom comparing algorithms
        if args.custom_algorithms:
            from ait_llm.compare.cmp_algorithm import register_custom_compare_algorithm

            for custom_compare_algorithm in args.custom_algorithms:
                register_custom_compare_algorithm(custom_compare_algorithm)

        op = OpChecker()
        logger.info(f"===================Opcheck start====================")
        op.start_test(args)
        logger.info(f"===================Opcheck end====================")


class ErrCheck(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--exec',
            dest="exec",
            required=True,
            type=safe_string,
            default='',
            help='Executable command that running acl-transformer model inference. '
                 'User is responsible for the safeness of the input command. '
                 "E.g. --exec 'bash run.sh patches/models/modeling_xxx.py'.")

        parser.add_argument(
            '--type',
            dest="type",
            nargs='+', # one or more
            choices=['overflow'],
            default=['overflow'],
            help="Types that perform different error detection tasks. "
                 "Multiple arguments will trigger all the providing functionalities."
        )

        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            type=check_output_path_legality,
            default='',
            help="Directory that stores the error information. If not provided, a default directory will be used."
        )

        parser.add_argument(
            '--exit',
            dest='exit',
            required=False,
            action='store_true',
            default=False,
            help="Flag determines whether to exit the program after detecting an error. Defaults to False."
        )

        parser.add_argument("-l", "--log-level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        set_log_level(args.log_level)
        process_error_check(args)       


class Transform(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        scenarios_info = [
            "[float atb to quant atb model] directory containing both cpp and h file",
            "[float atb to quant atb model] a single cpp file, will use the h file with a same name",
            "[torch to float atb model] directory containing config.json and py file for building transformers model",
        ]
        scenarios_info_str = "; ".join([f"{id}.{ii}" for id, ii in enumerate(scenarios_info, start=1)])
        
        parser.add_argument(
            "-s",
            "--source",
            type=check_input_path_legality,
            required=True,
            help="source path, could be:" + scenarios_info_str,
        )
        parser.add_argument(
            "--enable-sparse", action='store_true', help="[float atb to quant atb model] Enable trasforming to sparse-quant model"
        )
        parser.add_argument("-l", "--log-level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level")

    def handle(self, args, **kwargs) -> None:
        from ait_llm.transform.utils import get_transform_scenario, SCENARIOS

        set_log_level(args.log_level)
        scenario = get_transform_scenario(args.source)
        logger.info(f"Current scenario: {scenario}")
        if scenario == SCENARIOS.float_atb_to_quant_atb:
            from ait_llm.transform.float_atb_to_quant_atb import transform_quant

            transform_quant.transform_quant(source_path=args.source, enable_sparse=args.enable_sparse)
        elif scenario == SCENARIOS.torch_to_float_atb:
            from ait_llm.transform.torch_to_float_atb import transform_float

            transform_float.transform_float(source_path=args.source)
        else:
            message = f"Neither config.json + py or cpp found in {args.source}, not supported"
            logger.error(message)
            raise ValueError(message)


def get_cmd_instance():
    llm_help_info = "Large Language Model(llm) Debugger Tools."
    dump_cmd_instance = DumpCommand("dump", "Dump tool for ascend transformer boost", alias_name="dd")
    compare_cmd_instance = CompareCommand("compare", "Accuracy compare tool for large language model", alias_name="cc")
    opcheck_cmd_instance = OpcheckCommand("opcheck", "Operation check tool for large language model", alias_name='oo')
    errcheck_cmd_instance = ErrCheck("errcheck", "Error check tool for large language model.", alias_name='ee')
    transform_cmd_instance = Transform("transform", "Transform tool for large language model.")

    instances = [
        dump_cmd_instance, compare_cmd_instance, opcheck_cmd_instance, errcheck_cmd_instance, transform_cmd_instance
    ]
    return BaseCommand("llm", llm_help_info, instances)
