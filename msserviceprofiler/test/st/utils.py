import subprocess
import os
import re
import logging

COMMAND_SUCCESS = 0

def execute_cmd(cmd):
    logging.info('Execute command:%s' % " ".join(cmd))
    completed_process = subprocess.run(cmd, shell=False, stderr=subprocess.PIPE)
    if completed_process.returncode != COMMAND_SUCCESS:
        logging.error(completed_process.stderr.decode())
    return completed_process.returncode


def check_column_actual(actual_columns, expected_columns, context):
    """检查实际列名是否与预期列名一致"""
    for col in expected_columns:
        assert col in actual_columns, f"在 {context} 中未找到预期列名: {col}"


def check_row(df, row_index, column):
    """检查指定行和列的数据是否为数字"""
    try:
        value = df.at[row_index, column]
        # 尝试将值转换为数字
        float(value)
    except (ValueError, KeyError):
        assert False, f"在 {column} 列的第 {row_index} 行，值 {value} 不是有效的数字"