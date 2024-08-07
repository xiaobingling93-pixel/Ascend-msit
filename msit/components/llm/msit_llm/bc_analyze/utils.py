import os
from random import Random
from datetime import datetime, timedelta, timezone


class RandomNameSequence(object):
    characters = "abcdefghijklmnopqrstuvwxyz0123456789_"

    def __init__(self):
        self._rng = None
        self._rng_pid = None

    @property
    def rng(self):
        cur_pid = os.getpid()
        if cur_pid != getattr(self, '_rng_pid', None):
            self._rng = Random()
            self._rng_pid = cur_pid
        return self._rng
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return ''.join(self.rng.choices(self.characters, k=8))


def get_timestamp():
    cst_timezone = timezone(timedelta(hours=8))
    current_time = datetime.now(cst_timezone)
    return current_time.strftime("%Y%m%d%H%M%S")