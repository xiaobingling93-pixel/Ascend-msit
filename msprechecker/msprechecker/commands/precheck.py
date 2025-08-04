# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import argparse
from textwrap import dedent

from .base import CommandType
from .legacy import add_legacy_argument
from ..utils import ErrorSeverity


def setup_precheck_parser(subparsers):
    precheck_parser = subparsers.add_parser(
        CommandType.CMD_PRECHECK.value,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent('''\
            PRECHECK - Run a seires of validations and checks for different PD deployment.

            Mix Mode (default):
              - Validate environment, system, and HCCL settings.
              - Generate 'msprechecker_env.sh' if environment issues found.

            Disaggregation Mode:
              - Validate only user_config.json and mindie_env.json.
              - Skip environment/system/HCCL checks.
        '''),
        usage='msprechecker precheck [OPTIONS]',
        epilog=dedent('''\
            Examples:
              msprechecker precheck --rank-table hccl_8s_64p.json --weight-dir ./Llama3-70B                   # Full validation (default)
              msprechecker precheck --user-config-path /config/user.json --mindie-env-path /env/mindie.json   # Disaggregation mode
        '''),
        help="Run comprehensive system validation for different PD deployment scenarios"
    )

    add_legacy_argument(precheck_parser)

    pd_disagg_group = precheck_parser.add_argument_group("PD Disaggregation Options")
    pd_disagg_group.add_argument(
        "--user-config-path",
        metavar="",
        help="Path to the 'user_config.json' file for Kubernetes-based deployments."
    )
    pd_disagg_group.add_argument(
        "--mindie-env-path",
        metavar="",
        help="Path to the 'mindie_env.json' file for Kubernetes-based deployments."
    )

    pd_mix_group = precheck_parser.add_argument_group("PD Mixed Mode Options")
    pd_mix_group.add_argument(
        "--mies-config-path",
        metavar="",
        help="Path to the 'config.json' file for daemon-based deployments."
    )

    network_group = precheck_parser.add_argument_group("Network Options")
    network_group.add_argument(
        "--rank-table-path",
        metavar="",
        help="Path to the rank table file. Supports both A2 and A3 formats."
    )

    model_group = precheck_parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--weight-dir",
        metavar="",
        help="Directory path containing model weights."
    )

    stress_test_group = precheck_parser.add_argument_group("Stress Test Options")
    stress_test_group.add_argument(
        "--hardware",
        action="store_true",
        default=False,
        help="Enable hardware stress testing. Default: False."
    )
    stress_test_group.add_argument(
        "--threshold",
        type=int,
        choices=range(0, 101),
        default=20,
        metavar="0-100",
        help="Set the failure threshold percentage (0-100). Default: 20."
    )

    custom_group = precheck_parser.add_argument_group("Custom Validation Options")
    custom_group.add_argument(
        "--custom-config-path",
        metavar="",
        help="Path to a custom validation rules configuration file."
    )
    custom_group.add_argument(
        "-l", "--severity-level",
        metavar="",
        choices=[ErrorSeverity.ERR_LOW, ErrorSeverity.ERR_MEDIUM, ErrorSeverity.ERR_HIGH],
        default=ErrorSeverity.ERR_LOW,
        type=ErrorSeverity,
        help="Report only issues with the specified severity level or higher. Default: low."
    )

    return precheck_parser
