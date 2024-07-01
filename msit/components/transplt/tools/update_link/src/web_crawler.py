import time

from selenium import webdriver
from selenium.common import NoSuchElementException, TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


ASCEND_URL_PREFIX = "https://www.hiascend.com"


class WebCrawler:
    def __init__(self):
        self._init_driver()
        self._crawled_first_url = False

    def _init_driver(self):
        options = webdriver.ChromeOptions()
        prefs = {
            'profile.default_content_setting_values': {
                'images': 2,
            }
        }
        options.add_experimental_option('prefs', prefs)
        svc = Service("./driver/chromedriver.exe")
        self.driver = webdriver.Chrome(service=svc, options=options)

    def _check_if_throttled(self):
        try:
            WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "topic-title")))
            self.driver.find_element(By.CLASS_NAME, "topic-title")
        except (TimeoutException, NoSuchElementException):
            return True
        return False

    def _try_wait_for_section_loaded(self):
        try:
            WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "sectiontitle")))
        except TimeoutException:
            pass

    def try_get_api_class_name(self):
        _tmp_class_name = ""
        try:
            title = self.driver.find_element(By.CLASS_NAME, "o-breadcrumb").text.strip()
            names = title.split("\n")
            if len(names) > 2 and names[-2].strip().isascii():
                _tmp_class_name = names[-2].strip()
        except NoSuchElementException:
            _tmp_class_name = ""
        return _tmp_class_name

    def open_ascend_url(self, target_url):
        """
        :param target_url:
        :return: 1) if opening url succeeded. 2) if is throttled
        """

        if not target_url.startswith(ASCEND_URL_PREFIX):
            raise ValueError(f"not support to open non-ascend url {target_url}")
        self.driver.get(target_url)
        if not self._crawled_first_url:
            self._crawled_first_url = True
            time.sleep(8)
        else:
            time.sleep(4)

        if self._check_if_throttled():
            return False, True

        self._try_wait_for_section_loaded()

        return True, False

    def reinit_crawler(self):
        if self.driver is not None:
            self.driver.close()
            self._init_driver()

    def close(self):
        self.driver.close()
