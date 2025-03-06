# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

import json
import argparse
import ast
import re
import csv
import pandas as pd
from tabulate import tabulate

from msit_llm.common.log import logger
from components.utils.file_open_check import ms_open

service_parameters = ['temperature', 'top_k', 'top_p', 'do_sample', 'seed', 'repetition_penalty', 'watermark',
                      'frequency_penalty', 'presence_penalty', 'length_penalty', 'ignore_eos']


def create_difference_dict(req_order, param, file1_value="N/A", file2_value="N/A"):
    if req_order == -1:
        return {"param": param, "file1_value": file1_value, "file2_value": file2_value}
    else:
        return {"req_order": req_order, "param": param, "file1_value": file1_value, "file2_value": file2_value}


def parse_parameter_line(line, params, current_request, file_line_number):
    extract_params = line.strip().split(':', 1)   # 防止value值中含冒号的情况
    if len(extract_params) == 2:
        key_part, value_part = extract_params   # 将参数分为key和value两部分，例如"temperature"和[0.69999]

        key = key_part.strip().strip('"')    # 去掉key中多余的空格和""
        value = value_part.strip().strip('[],')  # 去掉value中多余的空格和[]

        if value == 'null':
            value = 'None'

        try:
            parsed_value = ast.literal_eval(value)
            if key == 'do_sample' and parsed_value is not None:
                parsed_value = bool(parsed_value)
            params[key] = parsed_value
        except (ValueError, SyntaxError) as e:
            logger.error(f"Value parsing error at line {file_line_number} (request {current_request}): {e}")
    elif '{' not in extract_params and '}' not in extract_params and extract_params != ['']:  # 过滤{ }和空行
        logger.warning(
            f"Unexpected parameter format at line {file_line_number}: "
            f"'{line.strip()}' (request: {current_request})"
        )
    else:
        pass

    
def extract_log_parameters(log_file_path):
    """
    Extract sampling parameters from the log file and return a dictionary.

    Parameters:
        log_file_path (str): Path to the log file.

    Returns:
        dict: A dictionary containing all the request parameters, structured as {request_id_counter: parameters_dict}.
    """
    result_dict = {}

    # 定义正则表达式
    start_pattern = re.compile(
        r'\[endpoint\] Sampling parameters for request id: (\S+)'  
        r'|'  
        r'Sampling parameters for trace ids \[([^]]*)\]:'  
    )

    current_request = None
    brace_count = 0
    flag_begin = False
    match_cache = ""

    with ms_open(log_file_path, 'r', encoding='utf-8') as f:
        request_id_counter = 1
        for file_line_number, line in enumerate(f, 1):
            # 匹配参数块开始行
            start_match = start_pattern.search(line)
            if start_match:
                # 处理未完成的前一个请求
                if current_request is not None:
                    logger.error(f"Abandon incomplete block for request {current_request}")

                # 提取请求标识符（request_id 或 trace_id）
                current_request = start_match.group(1) or start_match.group(2)
                brace_count = 0
                match_cache = start_match
                continue

            if match_cache and '{' in line:
                params = {}
                flag_begin = True
            # 处理参数块内容
            if flag_begin:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                parse_parameter_line(line, params, current_request, file_line_number)
                if brace_count == 0:
                    result_dict[request_id_counter] = params
                    request_id_counter += 1
                    flag_begin = False
                    match_cache = ""
                    current_request = None
    return result_dict


def extract_txt_parameters(log_file_path):
    """
    Extract sampling parameters from the log file and return a dictionary.

    Parameters:
        log_file_path (str): Path to the log file.

    Returns:
        dict: A dictionary containing all the request parameters, structured as {request_id_counter: parameters_dict}.
    """
    with ms_open(log_file_path, 'r') as file:
        param_str = file.read()

    # **第一步**：拆分每个 `request_id: SamplingParams(...)` 对
    # 假设每个 request_id 和 SamplingParams 一对都在一行或一段中
    lines = param_str.split("\n")

    param_dict = {}
    request_id_counter = 1  # 用来跟踪每个 request_id 的顺序
    request_id_cache = []

    for line in lines:
        # 找到包含 `request_id: SamplingParams(...)` 的行
        if ':' in line:
            # **第二步**：提取 `request_id` 和 `SamplingParams(...)` 部分
            request_id, param_str = line.split(":", 1)
            if request_id in request_id_cache:
                continue
            request_id_cache.append(request_id)

            # **第三步**：提取do_sample参数; 去掉 "SamplingParams" 这个类名，只保留括号内部的部分
            do_sample = param_str.split(')')[1].split(':')
            param_str = param_str[param_str.find("(") + 1: param_str.rfind(")")]

            # **第四步**：逐字符解析，确保不会拆开 `[]`, `{}`, `()`，并把参数分割成键值对
            params = []
            current = ""
            depth = 0  # 用于跟踪嵌套深度（如 `[]`, `{}`, `()`）

            for char in param_str:
                if char in "([{":
                    depth += 1  # 进入嵌套
                elif char in ")]}":
                    depth -= 1  # 退出嵌套
                else:
                    pass

                if char == "," and depth == 0:  # 只有在 **不在嵌套** 时才能切割
                    params.append(current.strip())
                    current = ""
                else:
                    current += char

            # 处理最后一个参数
            if current:
                params.append(current.strip())

            # **第五步**：解析键值对并存入字典
            request_params = {}
            for pair in params:
                key, value = pair.split("=", 1)  # 只拆分第一个 `=`
                key = key.strip()
                value = value.strip()

                # 尝试将值转换为 Python 变量
                try:
                    parsed_value = ast.literal_eval(value)  # 解析 Python 数据类型（列表、None、True、False、数字）
                except (ValueError, SyntaxError):
                    parsed_value = value.strip("'")  # 处理普通字符串

                request_params[key] = parsed_value  # 存入字典
            if do_sample != ['']:
                request_params[do_sample[0]] = bool(do_sample[1])
            # **第六步**：使用 request_id_counter 作为 param_dict 的键
            param_dict[request_id_counter] = request_params
            request_id_counter += 1  # 增加 request_id_counter
    return param_dict


def compare_parameters(dict_input1, dict_input2, path="", depth=0):
    if depth > 5:
        raise RecursionError(f"The recursion depth exceeds the maximum limit (5). Current path:{path}")
    """比对两个字典"""
    differences = []

    # 遍历第一个字典的key
    for key in dict_input1:
        new_path = f"{path}.{key}" if path else key
        if key not in dict_input2:
            differences.append(create_difference_dict(-1, new_path, dict_input1[key], "N/A"))
        else:
            # 如果值是字典，则递归调用
            if isinstance(dict_input1[key], dict) and isinstance(dict_input2[key], dict):
                differences.extend(compare_parameters(dict_input1[key], dict_input2[key], new_path, depth + 1))
            # 如果值是列表，则比对列表
            elif isinstance(dict_input1[key], list) and isinstance(dict_input2[key], list):
                # 比较数组的长度
                if len(dict_input1[key]) != len(dict_input2[key]):
                    differences.append(create_difference_dict(-1, new_path, dict_input1[key], dict_input2[key]))
                else:
                    for i, (item1, item2) in enumerate(zip(dict_input1[key], dict_input2[key])):
                        if item1 != item2:
                            differences.append(create_difference_dict(-1, f"{new_path}[{i}]",
                                                                      dict_input1[key][i], dict_input2[key][i]))
            else:
                # 比较值是否一致
                if dict_input1[key] != dict_input2[key]:
                    differences.append(create_difference_dict(-1, new_path, dict_input1[key], dict_input2[key]))

    # 遍历第二个字典的key，找出遗漏的key
    for key in dict_input2:
        new_path = f"{path}.{key}" if path else key
        if key not in dict_input1:
            differences.append(create_difference_dict(-1, new_path, "N/A", dict_input2[key]))
    return differences


def compare_service_parameters(dict_input1, dict_input2, req_order, differences):
    for param in service_parameters:
        if param not in dict_input1 and param not in dict_input2:
            continue
        elif param not in dict_input1:
            differences.append(create_difference_dict(req_order, param, "N/A", dict_input2[param]))

        elif param not in dict_input2:
            differences.append(create_difference_dict(req_order, param, dict_input1[param], "N/A"))
        else:
            if dict_input1[param] != dict_input2[param]:
                differences.append(create_difference_dict(req_order, param, dict_input1[param], dict_input2[param]))


def compare_and_generate(input1_params, input2_params):
    differences = []
    if len(input1_params) != len(input2_params):
        logger.error('Please make sure that the request parameters are the same on both sides')
    else:
        for req_order, params in input1_params.items():
            compare_service_parameters(input2_params[req_order], params, req_order, differences)
        generate_report(differences)


def service_params_check(input1, input2):
    if input1.endswith('.json') and input2.endswith('.json'):
        """加载两个 JSON 文件并进行比对"""
        with ms_open(input1, "r", encoding="utf-8") as f_gpu, ms_open(input2, "r", encoding="utf-8") as f_npu:
            dict_input1 = json.load(f_gpu)
            dict_input2 = json.load(f_npu)
        differences = compare_parameters(dict_input1, dict_input2)
        generate_report(differences)
    elif input1.endswith('.txt') and input2.endswith('.log'):
        gpu_input = extract_txt_parameters(input1)
        npu_input = extract_log_parameters(input2)
        compare_and_generate(input1_params=npu_input, input2_params=gpu_input)
    elif input1.endswith('.log') and input2.endswith('.log'):
        npu_input1 = extract_log_parameters(input1)
        npu_input2 = extract_log_parameters(input2)
        compare_and_generate(npu_input1, npu_input2)
    else:
        logger.error('Please make sure that the type and format of the input file is correct')


def generate_report(differences, output_file="comparison_report.csv"):
    """生成比对报告并保存为CSV文件"""
    if not differences:
        logger.info(f"No differences found")
    else:
        df = pd.DataFrame(differences)
        df['file1_value'] = df['file1_value'].astype(str)
        df['file2_value'] = df['file2_value'].astype(str)
        df.to_csv(output_file, na_rep="None", index=False, encoding='utf-8')
        df = df.replace({"nan": "None"})
        table = tabulate(df, headers=df.columns, tablefmt="grid", missingval="None",
                         showindex=False, disable_numparse=True)
        logger.info(f"\n{table}")
        logger.info(f"Report saved to {output_file}")