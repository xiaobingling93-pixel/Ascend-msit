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
from tabulate import tabulate

from msit_llm.common.log import logger
from components.utils.file_open_check import ms_open

service_parameters = ['temperature', 'top_k', 'top_p', 'do_sample', 'seed', 'repetition_penalty', 'watermark',
                      'frequency_penalty', 'presence_penalty', 'length_penalty', 'ignore_eos']


def extract_log_parameters(log_file_path):
    """
    从日志文件中提取采样参数并返回字典

    参数：
        log_file_path (str): 日志文件路径

    返回：
        dict: 包含所有请求参数的字典，结构为{request_id: parameters_dict}
    """
    result_dict = {}

    # 定义正则表达式模式
    start_pattern = re.compile(
        r'\[endpoint\] Sampling parameters for request id: (\S+)'
    )

    current_request = None
    json_buffer = []
    brace_count = 0

    with ms_open(log_file_path, 'r', encoding='utf-8') as f:
        count_line = 1
        for line in f:
            # 匹配参数块开始行
            start_match = start_pattern.search(line)
            if start_match:
                current_request = start_match.group(1)
                brace_count = 0
                json_buffer = []
                continue

            # 处理参数块内容
            if current_request:
                # 统计大括号数量
                brace_count += line.count('{')
                brace_count -= line.count('}')

                # 收集JSON内容
                json_buffer.append(line)

                # 当大括号数量归零时尝试解析
                if brace_count == 0 and current_request:
                    try:
                        json_str = ''.join(json_buffer)
                        params = json.loads(json_str)
                        result_dict[count_line] = params
                        count_line += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"解析失败：{current_request}，错误：{e}")
                    finally:
                        current_request = None
                        json_buffer = []
                    # break
    return result_dict


def extract_txt_parameters(log_file_path):
    """
    从日志文件中提取采样参数并返回字典

    参数：
        log_file_path (str): 日志文件路径

    返回：
        dict: 包含所有请求参数的字典，结构为{request_id: parameters_dict}
    """
    with open(log_file_path, 'r') as file:
        param_str = file.read()

    # **第一步**：拆分每个 `request_id: SamplingParams(...)` 对
    # 假设每个 request_id 和 SamplingParams 一对都在一行或一段中
    lines = param_str.split("\n")

    param_dict = {}
    request_id_counter = 1  # 用来跟踪每个 request_id 的顺序

    for line in lines:
        # 找到包含 `request_id: SamplingParams(...)` 的行
        if ':' in line:
            # **第二步**：提取 `request_id` 和 `SamplingParams(...)` 部分
            request_id, param_str = line.split(":", 1)
            request_id = request_id.strip()  # 清理空白字符

            # **第三步**：去掉 "SamplingParams" 这个类名，只保留括号内部的部分
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

            # **第六步**：使用 request_id_counter 作为 param_dict 的键
            param_dict[request_id_counter] = request_params
            request_id_counter += 1  # 增加 request_id_counter
    return param_dict


def compare_parameters(dict_input1, dict_input2, path=""):
    """比对两个字典"""
    differences = []

    # 遍历第一个字典的key
    for key in dict_input1:
        new_path = f"{path}.{key}" if path else key
        if key not in dict_input2:
            differences.append({"key": new_path, "file1_value": dict_input1[key], "file2_value": "N/A"})
        else:
            # 如果值是字典，则递归调用
            if isinstance(dict_input1[key], dict) and isinstance(dict_input2[key], dict):
                differences.extend(compare_parameters(dict_input1[key], dict_input2[key], new_path))
            # 如果值是列表，则比对列表
            elif isinstance(dict_input1[key], list) and isinstance(dict_input2[key], list):
                # 比较数组的长度
                if len(dict_input1[key]) != len(dict_input2[key]):
                    differences.append({
                        "key": new_path, 
                        "file1_value": dict_input1[key], 
                        "file2_value": dict_input2[key]
                        })
                else:
                    for i, (item1, item2) in enumerate(zip(dict_input1[key], dict_input2[key])):
                        if item1 != item2:
                            differences.append({
                                "key": f"{new_path}[{i}]", 
                                "file1_value": dict_input1[key][i], 
                                "file2_value": dict_input2[key][i]
                                })
            else:
                # 比较值是否一致
                if dict_input1[key] != dict_input2[key]:
                    differences.append({
                        "key": new_path, 
                        "file1_value": dict_input1[key], 
                        "file2_value": dict_input2[key]
                        })

    # 遍历第二个字典的key，找出遗漏的key
    for key in dict_input2:
        new_path = f"{path}.{key}" if path else key
        if key not in dict_input1:
            differences.append({
                "key": new_path, 
                "file1_value": "N/A", 
                "file2_value": dict_input2[key]
                })

    return differences


def compare_service_parameters(dict_input1, dict_input2, req_order):
    differences = []
    for param in service_parameters:
        if param not in dict_input1 and param not in dict_input2:
            differences.append({
                "req_order": req_order, 
                "key": param, 
                "file1_value": "N/A", 
                "file2_value": "N/A"
                })
        elif param not in dict_input1:
            differences.append({
                "req_order": req_order, 
                "key": param, 
                "file1_value": "N/A", 
                "file2_value": dict_input2[param]})

        elif param not in dict_input2:
            differences.append({
                "req_order": req_order, 
                "key": param, 
                "file1_value": dict_input1[param], 
                "file2_value": "N/A"
                })
        else:
            if dict_input1[param] != dict_input2[param]:
                differences.append({
                    "req_order": req_order, 
                    "key": param, 
                    "file1_value": dict_input1[param], 
                    "file2_value": dict_input2[param]
                    })
    return differences


def compare_files(input1, input2):
    is_first = True
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
        if len(gpu_input) != len(npu_input):
            logger.error('Please make sure that the request parameters are the same on both sides')
        else:
            for req_order, params in npu_input.items():
                differences = compare_service_parameters(gpu_input[req_order], params, req_order)
                if is_first:
                    generate_report(differences, mode='w', is_multi=True)
                    is_first = False
                else:
                    generate_report(differences, mode='a', is_multi=True)
    elif input1.endswith('.log') and input2.endswith('.log'):
        npu_input1 = extract_log_parameters(input1)
        npu_input2 = extract_log_parameters(input2)
        if len(npu_input1) != len(npu_input2):
            logger.error('Please make sure that the request parameters are the same on both sides')
        else:
            for req_order, params in npu_input1.items():
                differences = compare_service_parameters(npu_input2[req_order], params, req_order)
                if is_first:
                    generate_report(differences, mode='w', is_multi=True)
                    is_first = False
                else:
                    generate_report(differences, mode='a', is_multi=True)
    else:
        logger.error('Please make sure that the type and format of the input file is correct')


def generate_report(differences, output_file="comparison_report.csv", mode='w', is_multi=False):
    """生成比对报告并保存为CSV文件"""
    if not differences:
        # 如果没有差异，返回一个绿色的提示并保存空报告
        with ms_open(output_file, mode=mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["No differences found!"])
        logger.info(f"No differences found")

    else:
        # 创建并写入 CSV 文件
        report = []
        with ms_open(output_file, mode=mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 写入表头
            if is_multi:
                writer.writerow(["req_order", "parameters", "value1", "value2"])
            else:
                writer.writerow(["parameters", "value1", "value2"])
            # 写入每条差异
            for diff in differences:
                if diff["file1_value"] is None:
                    diff["file1_value"] = "None"
                if diff["file2_value"] is None:
                    diff["file2_value"] = "None"
                # 写入每一行
                if is_multi:
                    writer.writerow([diff["req_order"], diff["key"], diff["file1_value"], diff["file2_value"]])
                    report.append([diff["req_order"], diff["key"], diff["file1_value"], diff["file2_value"]])
                else:
                    writer.writerow([diff["key"], diff["file1_value"], diff["file2_value"]])
                    report.append([diff["key"], diff["file1_value"], diff["file2_value"]])

        if is_multi:
            table = tabulate(report, headers=["req_order", "parameters", "value1", "value2"], tablefmt="grid")
        else:
            table = tabulate(report, headers=["parameters", "value1", "value2"], tablefmt="grid")
        logger.info(f"\n{table}")
    logger.info(f"Report saved to {output_file}")


def main():
    """主函数，处理命令行参数并调用比对逻辑"""
    parser = argparse.ArgumentParser(description="Compare two files and generate a report.")
    parser.add_argument("--input1", dest='input1', default='', type=str,
                        help="<Optional> Path of GPU config json file", required=True)
    parser.add_argument("--input2", dest='input2', default='', type=str,
                        help="<Optional> Path of NPU config json file", required=True)
    args = parser.parse_args()

    compare_files(args.input1, args.input2)




if __name__ == "__main__":
    main()