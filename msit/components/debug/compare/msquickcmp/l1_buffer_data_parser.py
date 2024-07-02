#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
"""
Function:
This class mainly involves the main function.
"""

import logging
import argparse
import os
import stat
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR


class L1BufferDataParser:
    """
    The class for l1 buffer data parser
    """
    NONE_ERROR = 0
    INVALID_PARAM_ERROR = 1
    UNKNOWN_ERROR = 2
    TWO_M = 2 * 1024 * 1024

    def __init__(self, args):
        self.dump_path = os.path.realpath(args.dump_path)
        self.output_path = os.path.realpath(args.output_path)
        self.offset = args.offset
        self.size = args.size

    def check_argument_valid(self):
        self._check_path_valid(self.dump_path, is_file=True)
        self._check_path_valid(self.output_path, is_file=False)
        if self.offset < 0 or self.offset >= self.TWO_M:
            logger.error('The offset (%d) is invalid, out of range [0, %d). Please check the offset.'
                  % (self.offset, self.TWO_M))
            sys.exit(self.INVALID_PARAM_ERROR)
        if self.size <= 0 or self.size > self.TWO_M - self.offset:
            logger.error('The size (%d) is invalid, out of range (0, %d). Please check the size.'
                  % (self.size, self.TWO_M - self.offset))
            sys.exit(self.INVALID_PARAM_ERROR)

    def parse(self):
        self.check_argument_valid()
        with open(self.dump_path, 'rb') as l1_buffer_data_file:
            if self.offset > 0:
                l1_buffer_data_file.read(self.offset)
            data = l1_buffer_data_file.read(self.size)
            output_file_path = os.path.join(self.output_path,
                                            "%s.%d.%d" % (os.path.basename(self.dump_path), self.offset, self.size))
            with os.fdopen(os.open(output_file_path, OPEN_FLAGS, OPEN_MODES),
                           "wb") as output_file:
                output_file.write(data)
            logger.info("The l1 buffer data for [%d, %d) has been saved in %s."
                  % (self.offset, self.offset + self.size, output_file_path))

    def _check_path_valid(self, path, is_file):
        if not os.path.exists(path):
            logger.error('The path "%s" does not exist. Please check the path.' % path)
            sys.exit(self.INVALID_PARAM_ERROR)
        if not os.access(path, os.R_OK):
            logger.error('You do not have permission to read the path "%s". Please check the path.' % path)
            sys.exit(self.INVALID_PARAM_ERROR)
        if is_file:
            if not os.path.isfile(path):
                logger.error('The path "%s" is not a file. Please check the path.' % path)
                sys.exit(self.INVALID_PARAM_ERROR)
            file_size = os.path.getsize(path)
            if file_size != self.TWO_M:
                logger.error('The l1 buffer data size (%d) is not %d for "%s". Please check the path.'
                      % (file_size, self.TWO_M, path))
                sys.exit(self.INVALID_PARAM_ERROR)
        else:
            if not os.path.isdir(path):
                logger.error('The path "%s" is not a directory. Please check the path.' % path)
                sys.exit(self.INVALID_PARAM_ERROR)
            if not os.access(path, os.W_OK):
                logger.error('You do not have permission to write the path "%s". Please check the path.' % path)
                sys.exit(self.INVALID_PARAM_ERROR)


def _parser_argument(parser):
    parser.add_argument("-d", "--dump-path", dest="dump_path", default="",
                        help="<Required> The l1 buffer data path", required=True)
    parser.add_argument("-o", "--offset", dest="offset", type=int, default=0,
                        help="<Optional> The offset of the data. the default value is 0.")
    parser.add_argument("-s", "--size", dest="size", type=int,
                        help="<Required> The size of the data", required=True)
    parser.add_argument("-out", "--output-path", dest="output_path", default="", help="<Optional> The output path")


def main():
    """
    Function Description:
        main process function
    Exception Description:
        exit the program when an AccuracyCompare Exception  occurs
    """
    parser = argparse.ArgumentParser()
    _parser_argument(parser)
    args = parser.parse_args(sys.argv[1:])
    try:
        L1BufferDataParser(args).parse()
    except Exception as ex:
        logger.error('Failed to parse the l1 buffer data. %s' % str(ex))
        sys.exit(L1BufferDataParser.UNKNOWN_ERROR)
    sys.exit(L1BufferDataParser.NONE_ERROR)


if __name__ == '__main__':
    main()
