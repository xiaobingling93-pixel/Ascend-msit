import os
import re
from typing import List, Tuple, Set


class FileCollector(object):
    def __init__(self, pattern: re.Pattern, max_iter=100) -> None:
        self.pattern = pattern
        self.max_iter = max_iter

        self._validate_param()

    def _validate_param(self):
        if not isinstance(self.pattern, re.Pattern):
            raise ValueError

        if not isinstance(self.max_iter, int):
            raise ValueError

    def _collect(self, dir_path: str) -> Set:
        res = set()

        files = os.listdir(dir_path)

        if len(files) > self.max_iter:
            raise RuntimeError

        for file_path in files:
            if self.pattern.match(file_path):
                res.add(file_path)

        return res


    def collect_pairs(self, dir_path_a: str, dir_path_b: str) -> List[Tuple]:
        file_set_a = self._collect(dir_path_a)
        file_set_b = self._collect(dir_path_b)

        intersection = file_set_a & file_set_b

        return [
            (os.path.join(dir_path_a, file_path), os.path.join(dir_path_b, file_path))
            for file_path in intersection
        ]

