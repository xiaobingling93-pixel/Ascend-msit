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
    run_parser = subparsers.add_parser('run', help='Run rule validations')
    run_parser.add_argument('rule', type=validate_args(Rule.input_file_read), help='Rule file to validate')
    run_parser.add_argument(
        '--contexts', '-C', nargs='*', help='Context variables in name:path format (or name:path@type).'
    )
    run_parser.add_argument(
        '--configs', '-c',
        nargs='*',
        help='Config files in name:path format (or name:path@type). Use @json or @yaml to force parse type'
    )
    run_parser.add_argument(
        '-co', '--collect-only', action='store_true', help='Show which rules will run, do not execute'
    )
    run_parser.add_argument(
        '-x', '--fail-fast', action='store_true', dest='failfast', help='Stop on first failure/error'
    )
    run_parser.add_argument(
        '-v', '--verbose', action='store_true', dest='verbose', help='Verbose: show each test name and result'
    )
    run_parser.add_argument(
        '-s', '--severity', 
        choices=['info', 'warning', 'error'],
        default='info',
        help='Minimum severity to run: info (all), warning (no info), error (errors only)'
    )

    inspect_parser = subparsers.add_parser('inspect', help='Inspect rule requirements')
    inspect_parser.add_argument('rule', type=validate_args(Rule.input_file_read), help='Rule file to inspect')
    inspect_parser.add_argument('--format', '-f', choices=['text', 'json'], default='text', help='Output format')

    