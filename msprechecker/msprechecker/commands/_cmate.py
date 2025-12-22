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

from msguard import validate_args, Rule


def setup_cmate_parser(subparsers):
    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Execute configuration validation against specified rules',
        description='Run comprehensive validation of configurations using the specified rule set.'
    )

    run_parser.add_argument(
        'rule',
        type=validate_args(Rule.input_file_read),
        help='Path to the rule definition file (CMATE format)'
    )

    run_parser.add_argument(
        '--configs', '-c',
        nargs='*',
        help=(
            "Configuration files to validate, specified as '<name>:<path>' or '<name>:<path>@<parse-type>'.\n"
            "  - <name>: Configuration identifier defined in the rule file\n"
            "  - <path>: File system path to the configuration file\n"
            "  - <parse-type>: (Optional) Parsing method ('json', 'yaml', 'yml')\n"
            "Note: For 'env' type targets, the <path> component is optional and will be discarded if provided."
        )
    )

    run_parser.add_argument(
        '--contexts', '-C',
        nargs='*',
        help=(
            "Context variables required for rule execution, specified as '<name>:<value>'.\n"
            "  - <name>: Context identifier defined in the rule file\n"
            "  - <value>: Value to assign to the context variable\n"
            "Note: Unquoted numbers are parsed as integers. Use quotes for string values: '2'."
        )
    )

    run_parser.add_argument(
        '-co', '--collect-only',
        action='store_true',
        help="Display the list of rules that would be executed without actually running them"
    )

    run_parser.add_argument(
        '-x', '--fail-fast',
        action='store_true',
        dest='failfast',
        help="Stop execution immediately upon encountering the first failure or error"
    )

    run_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help="Enable verbose output, displaying detailed test names and individual results"
    )

    run_parser.add_argument(
        '-s', '--severity',
        choices=['info', 'warning', 'error'],
        default='info',
        help=(
            "Minimum severity level for rule execution:\n"
            "  - info: Execute all checks (default)\n"
            "  - warning: Execute only warning and error checks\n"
            "  - error: Execute only error checks"
        )
    )

    # Inspect command
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Display detailed information about rule requirements',
        description='Inspect and display the configuration targets, contexts, and requirements defined in a rule file.'
    )

    inspect_parser.add_argument(
        'rule',
        type=validate_args(Rule.input_file_read),
        help='Path to the rule definition file (CMATE format)'
    )

    inspect_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help=(
            "Output format for the inspection results:\n"
            "  - text: Human-readable text format (default)\n"
            "  - json: Structured JSON format for programmatic processing"
        )
    )
