# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import pathlib

import click
from click_aliases import ClickAliasedGroup
from click.exceptions import UsageError

from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer, InferTestConfig, BigKernelConfig
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from components.debug.common import logger
from auto_optimizer.common.utils import check_output_model_path
from auto_optimizer.common.click_utils import optimize_onnx, CONTEXT_SETTINGS, \
    FormatMsg, cli_eva, list_knowledges

from auto_optimizer.options import (
    arg_path,
    arg_input,
    arg_output,
    arg_start,
    arg_end,
    opt_check,
    opt_optimizer,
    opt_recursive,
    opt_verbose,
    opt_soc,
    opt_device,
    opt_big_kernel,
    opt_attention_start_node,
    opt_attention_end_node,
    opt_infer_test,
    opt_loop,
    opt_threshold,
    opt_input_shape,
    opt_input_shape_range,
    opt_dynamic_shape,
    opt_output_size,
    opt_processes,
    opt_subgraph_input_shape,
    opt_subgraph_input_dtype,
)


@click.group(cls=ClickAliasedGroup, context_settings=CONTEXT_SETTINGS,
             short_help='Modify ONNX models, and auto optimizer onnx models.',
             no_args_is_help=True)
def cli() -> None:
    '''main entrance of auto optimizer.'''
    pass


@cli.command('list', short_help='List available Knowledges.', context_settings=CONTEXT_SETTINGS)
def command_list() -> None:
    list_knowledges()


@cli.command(
    'evaluate',
    aliases=['eva'],
    short_help='Evaluate model matching specified knowledges.',
    context_settings=CONTEXT_SETTINGS
)
@arg_path
@opt_optimizer
@opt_recursive
@opt_verbose
@opt_processes
def command_evaluate(
    path: pathlib.Path,
    optimizer: GraphOptimizer,
    recursive: bool,
    verbose: bool,
    processes: int,
) -> None:
    path_ = pathlib.Path(path.decode()) if isinstance(path, bytes) else path
    cli_eva(path_, optimizer, recursive, verbose, processes)


@cli.command(
    'optimize',
    aliases=['opt'],
    short_help='Optimize model with specified knowledges.',
    context_settings=CONTEXT_SETTINGS
)
@arg_input
@arg_output
@opt_optimizer
@opt_big_kernel
@opt_attention_start_node
@opt_attention_end_node
@opt_infer_test
@opt_soc
@opt_device
@opt_loop
@opt_threshold
@opt_input_shape
@opt_input_shape_range
@opt_dynamic_shape
@opt_output_size
def command_optimize(
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    optimizer: GraphOptimizer,
    infer_test: bool,
    big_kernel: bool,
    attention_start_node: str,
    attention_end_node: str,
    soc: str,
    device: int,
    loop: int,
    threshold: float,
    input_shape: str,
    input_shape_range: str,
    dynamic_shape: str,
    output_size: str
) -> None:
    # compatibility for click < 8.0
    input_model_ = pathlib.Path(input_model.decode()) if isinstance(input_model, bytes) else input_model
    output_model_ = pathlib.Path(output_model.decode()) if isinstance(output_model, bytes) else output_model
    if input_model_ == output_model_:
        logger.warning('output_model is input_model, refuse to overwrite origin model!')
        return

    if big_kernel:
        big_kernel_config = BigKernelConfig(
            attention_start_node=attention_start_node,
            attention_end_node=attention_end_node
        )
    else:
        big_kernel_config = None

    config = InferTestConfig(
        converter='atc',
        soc=soc,
        device=device,
        loop=loop,
        threshold=threshold,
        input_shape=input_shape,
        input_shape_range=input_shape_range,
        dynamic_shape=dynamic_shape,
        output_size=output_size,
    )
    applied_knowledges = optimize_onnx(
        optimizer=optimizer,
        input_model=input_model_,
        output_model=output_model_,
        infer_test=infer_test,
        config=config,
        big_kernel_config=big_kernel_config
    )
    if infer_test:
        logger.info('=' * 100)
    if applied_knowledges:
        logger.info('Result: Success')
        logger.info('Applied knowledges: ')
        for knowledge in applied_knowledges:
            logger.info(f'  {knowledge}')
        logger.info(f'Path: {input_model_} -> {output_model_}')
    else:
        logger.info('Result: Unable to optimize, no knowledges matched.')
    if infer_test:
        logger.info('=' * 100)


@cli.command(
    'extract',
    aliases=['ext'],
    short_help='Extract subgraph from onnx model.',
    context_settings=CONTEXT_SETTINGS
)
@arg_input
@arg_output
@arg_start
@arg_end
@opt_check
@opt_subgraph_input_shape
@opt_subgraph_input_dtype
def command_extract(
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    start_node_names: str,
    end_node_names: str,
    is_check_subgraph: bool,
    subgraph_input_shape: str,
    subgraph_input_dtype: str
) -> None:
    if input_model == output_model:
        logger.warning('output_model is input_model, refuse to overwrite origin model!')
        return
    output_model_path = output_model.as_posix()
    if not check_output_model_path(output_model_path):
        return

    # parse start node names and end node names
    start_nodes = [node_name.strip() for node_name in start_node_names.split(',')]
    end_nodes = [node_name.strip() for node_name in end_node_names.split(',')]

    onnx_graph = OnnxGraph.parse(input_model.as_posix())
    try:
        onnx_graph.extract_subgraph(
            start_nodes, end_nodes,
            output_model_path, is_check_subgraph,
            subgraph_input_shape, subgraph_input_dtype
        )
    except ValueError as err:
        logger.error(err)


if __name__ == "__main__":
    UsageError.show = FormatMsg.show
    cli()
