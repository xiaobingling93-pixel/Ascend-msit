from abc import ABC, abstractmethod


class BaseComparator(ABC):
    SUPPORTED_EXTENSIONS = []
    
    def __init__(self, out_db_conn, excel_writer):
        self.out_db_conn = out_db_conn
        self.excel_writer = excel_writer
    
    @classmethod
    def supports(cls, file_extension):
        return file_extension in cls.SUPPORTED_EXTENSIONS
    
    @abstractmethod
    def process(self, file_a, file_b):
        pass