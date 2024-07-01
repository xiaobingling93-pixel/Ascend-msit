import os.path
import time

from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
import pandas as pd

from .logger import logger
from .util import open_excel


class AscendApiCrawler:
    def __init__(self, crawler, url_template, start_index, end_index, output_excel="output.xlsx"):
        self.url_template = url_template
        self.is_crawling_acl = True if "cann" in url_template else False
        self.start_index = start_index
        self.end_index = end_index
        self._crawled_data = []
        self._crawled_api_index = 1
        self._cur_url = ""
        self._tmp_class_name = ""
        self._failed_url = []
        self._is_first = True
        self.output_excel = output_excel
        self.crawler = crawler
        self.sleep_time = 30

    def _crawl_ascend_apis(self, target_url, url_id):
        logger.info(f"crawling {target_url}")
        self._cur_url = target_url
        self._tmp_class_name = ""

        success, is_throttled = self.crawler.open_ascend_url(target_url)

        if is_throttled:
            logger.error("reached throttle limit!!")
            return False, True

        try:
            self._tmp_class_name = self.crawler.try_get_api_class_name()

            success = self._try_parse_func_api(url_id)
            if success:
                return True, False

            success = self._try_parse_nn_interface_api(url_id)
            if success:
                return True, False

            success = self._try_parse_struct_api(url_id)

            if not success:
                logger.warning(f"unable to find api prototype, skipping {self._cur_url}")

            return True, False
        except NoSuchElementException as e:
            logger.error(f"crawl_ascend_apis failed on {target_url}, error is: {e}")
            self._failed_url.append(target_url)
            return False, False

    def _try_parse_struct_api(self, url_id):
        try:
            api_name = self.crawler.driver.find_element(By.CLASS_NAME, "topic-title").text.strip()
            struct_prototype = self.crawler.driver.find_element(By.CLASS_NAME, "code-box").text.strip()
            self._add_to_crawled_data(api_name, struct_prototype, url_id)
            return True
        except NoSuchElementException:
            return False

    def _try_parse_nn_interface_api(self, url_id):
        success = False
        sections = self.crawler.driver.find_elements(By.CLASS_NAME, "section")
        for section in sections:
            title = section.find_element(By.CLASS_NAME, "sectiontitle").text.strip()
            if not title.startswith("acl"):
                continue

            api_name = title.strip()

            prototype_section = section
            section_id = prototype_section.get_attribute("id")

            prototype = prototype_section.find_element(By.XPATH, f"//div[@id='{section_id}']/ul/li")
            prototype_str = prototype.text.strip().split("：")[-1].strip()

            self._add_to_crawled_data(api_name, prototype_str, url_id)
            success = True

        return success

    def _add_to_crawled_data(self, api_name, prototype_str, url_id):
        if len(self._crawled_data) > 0 and self._crawled_data[-1][3] == api_name:
            logger.info(f"already crawled api: {api_name}, skipping")
            return

        fixed_api_name = api_name
        # ACL API都是函数形式的接口，没有类成员函数接口
        if not self.is_crawling_acl and len(self._tmp_class_name) > 0:
            fixed_api_name = f"{self._tmp_class_name}::{api_name}"
        logger.info("### %04d ### %04d ### %-20s ### %-20s ### %-60s" %
                    (self._crawled_api_index, url_id, api_name, fixed_api_name, prototype_str))
        self._crawled_data.append(
            [self._crawled_api_index, url_id, self._cur_url, api_name, fixed_api_name, prototype_str])
        self._crawled_api_index += 1

    def _try_parse_func_api(self, url_id):
        prototype_section = None
        api_name = self.crawler.driver.find_element(By.CLASS_NAME, "topic-title").text.strip()

        sections = self.crawler.driver.find_elements(By.CLASS_NAME, "section")
        for section in sections:
            title = section.find_element(By.CLASS_NAME, "sectiontitle").text.strip()
            if title == "函数原型":
                prototype_section = section
                break

        if not prototype_section:
            return False

        section_id = prototype_section.get_attribute("id")

        if self.is_crawling_acl:
            prototypes = prototype_section.find_elements(By.XPATH, f"//div[@id='{section_id}']/p")
            prototype_str_list = []
            for prototype in prototypes:
                prototype_str_list.append(prototype.text.strip())
            prototype_str = " ".join(prototype_str_list)
            self._add_to_crawled_data(api_name, prototype_str, url_id)
        else:
            prototypes = prototype_section.find_elements(By.XPATH, f"//div[@id='{section_id}']/div")
            for prototype in prototypes:
                prototype_str = prototype.text.strip()
                self._add_to_crawled_data(api_name, prototype_str, url_id)

        return True

    def _init_output_excel_file(self):
        sheet_name = self._get_result_sheet_name()
        excel = pd.ExcelWriter(self.output_excel, mode='w')
        df = pd.DataFrame()
        df.to_excel(excel, sheet_name=sheet_name)
        excel.close()

    def _write_to_excel(self):
        if not os.path.exists(self.output_excel):
            self._init_output_excel_file()

        df = pd.DataFrame(self._crawled_data,
                          columns=[
                              "ID",
                              "URL_ID",
                              "URL",
                              "Function Name",
                              "Fixed Function Name",
                              "Function Prototype",
                          ])
        sheet_name = self._get_result_sheet_name()

        excel = open_excel(self.output_excel)
        df.to_excel(excel, sheet_name=sheet_name, index=False)
        excel.close()

    def _get_result_sheet_name(self):
        if self.is_crawling_acl:
            sheet_name = "acl_apis"
        else:
            sheet_name = "mxbase_apis"
        return sheet_name

    def _close(self):
        self.crawler.close()

    def _do_sleep(self):
        logger.error(f"sleep for {self.sleep_time} sec!!")
        time.sleep(self.sleep_time)
        logger.error("sleep finished!!")

    def start_crawling(self):
        error_count = 0
        max_error_count = 5

        index = self.start_index
        while index <= self.end_index:
            url = self.url_template % index
            try:
                crawl_success, need_sleep = self._crawl_ascend_apis(url, index)
            except Exception as err:
                self._failed_url.append(url)
                logger.error(f"crawling error occurred, error is: {err}")
                crawl_success, need_sleep = False, True

            if crawl_success:
                index += 1
                if index % 20 == 0:
                    self._write_to_excel()
            elif need_sleep:
                error_count += 1
                self._do_sleep()

                if error_count > 0 and error_count % max_error_count == 0:
                    logger.error(f"crawling error occurred {error_count} times, restart crawler!!!!!!!!!!!!!!!!!")
                    self.crawler.reinit_crawler()

        self._write_to_excel()
        self._close()
        logger.info("failed url:")
        for url in self._failed_url:
            logger.info(url)
