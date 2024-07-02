import json
import os
import sys

from src.excel_updater import ConfigExcelUpdater
from src.ascend_api_crawler import AscendApiCrawler
from src.web_crawler import WebCrawler
from src import logger
from src.util import check_permission


def print_usage_and_exit():
    print(f"usage: python3 update_link.py <original_excel_dir> config.json [MODE]")
    print(f"usage: valid [MODE] value is 1, 2, 3")
    exit(-1)


def crawl_ascend_lib_apis(config, output_excel):
    url_template = config.get("url_template")
    version = config.get("version")
    url_template = url_template.replace("<version>", version)
    start_index = config.get("start_index")
    end_index = config.get("end_index")
    base_crawler = WebCrawler()
    ascend_crawler = AscendApiCrawler(
        base_crawler, url_template, start_index=start_index, end_index=end_index, output_excel=output_excel)
    ascend_crawler.start_crawling()


def main():
    if len(sys.argv) < 3:
        print_usage_and_exit()

    src_dir = os.path.realpath(sys.argv[1])
    config_file = os.path.realpath(sys.argv[2])

    mode = 3
    if len(sys.argv) == 4:
        mode = sys.argv[3]

    try:
        mode = int(mode)
        if mode not in (1, 2, 3):
            print(f"mode value {mode} is not valid")
            print_usage_and_exit()
    except ValueError:
        print(f"mode value {mode} is not valid")
        print_usage_and_exit()

    if not check_input(config_file, src_dir):
        print_usage_and_exit()

    new_api_output_excel = "ascend_apis.xlsx"

    with open(config_file, encoding="utf-8") as file:
        config_dict = json.load(file)

    log_level = config_dict.get("log_level", "info")
    logger.set_logger_level(log_level)

    if mode & 1:
        crawl_config = config_dict.get("crawl_config")
        for config_name, config in crawl_config.items():
            crawl_ascend_lib_apis(config, new_api_output_excel)

    if mode & 2:
        base_crawler = WebCrawler()
        updater = ConfigExcelUpdater(base_crawler, src_dir, new_api_output_excel)
        updater.update_excels()


def check_input(config_file, src_dir):
    if not check_permission(config_file):
        return False

    if not check_permission(src_dir):
        return False

    for path, _, file_list in os.walk(src_dir):
        for f in file_list:
            file_path = os.path.join(path, f)
            if not check_permission(file_path):
                return False

    return True


if __name__ == "__main__":
    main()
