import os

import pandas as pd

from .logger import logger


def check_permission(file):
    if not os.path.exists(file):
        logger.error(f"path: {file} not exist, please check if file or dir is exist")
        return False
    if os.path.islink(file):
        logger.error(f"path :{file} is a soft link, not supported, please import file(or directory) directly")
        return False
    return True


def open_excel(excel_path):
    try:
        excel = pd.ExcelWriter(excel_path, engine="openpyxl", mode='a', if_sheet_exists='overlay')
    except ValueError:
        excel = pd.ExcelWriter(excel_path, engine="openpyxl", mode='a', if_sheet_exists='replace')
    return excel
