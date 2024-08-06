import os

import numpy as np
import pandas as pd

from .utils import RandomNameSequence, get_timestamp
from ..common.log import logger


class Synthesizer(object):
    """"""
    HEADER = ['queries', 'input_token_ids', 'output_token_ids', 'passed']
    
    def __init__(self, *, queries=None, input_token_ids=None, output_token_ids=None, passed=None) -> None:
        """"""
        self._info = dict(
            zip(self.HEADER, (np.array([], dtype=object), ) * len(self.HEADER))
        )

        self.from_args(
            queries=queries,
            input_token_ids=input_token_ids,
            output_token_ids=output_token_ids,
            passed=passed
        )

    def _update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                modified_value = self._sanitize_value(value)
                self._info[key] = np.append(self._info[key], modified_value, axis=0)

    def _sanitize_value(self, value):
        # str is a iterable
        if isinstance(value, str) or not hasattr(value, '__iter__'):
            return np.array([str(value)], dtype=object)

        return np.fromiter((str(item) for item in value), dtype=object)
    
    def from_args(self, *, queries=None, input_token_ids=None, output_token_ids=None, passed=None):
        """"""
        self._update_attributes(
            queries=queries,
            input_token_ids=input_token_ids,
            output_token_ids=output_token_ids,
            passed=passed
        )

    @staticmethod
    def from_cmd(command) -> str:
        """"""
        import subprocess
        import shlex

        env = os.environ.copy()
        patcher_folder = os.path.join(os.path.dirname(__file__), 'patcher')
        if not os.path.exists(patcher_folder):
            logger.error("Directory '%s' not found, please try to reinstall the latest msit", patcher_folder)
            raise FileNotFoundError

        env['PYTHONPATH'] = patcher_folder + ':' + env.get('PYTHONPATH', '')

        temp_dir_name = next(RandomNameSequence())
        env['MSIT_TEMP_DIR_NAME'] = temp_dir_name

        split_command = shlex.split(command)

        KNOWN_INVALID_COMMAND = {
            "rm", "mv", "mkfs", "dd",
            "chown", "chmod",
            "shutdown", "reboot",
            "curl", "wget",
        }

        if split_command[0] in KNOWN_INVALID_COMMAND:
            logger.error("Invalid command '%s'", split_command)
            raise ValueError

        logger.debug("Split command %s", split_command)

        try:
            child_process = subprocess.Popen(
                split_command,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            logger.error("'%s': command not found, please run it separately first before using 'from_cmd'", command)
            raise

        logger.info("Command '%s' is running, all the outs and errors are suppressed", command)

        if child_process.wait() != 0:
            logger.error("Failed to run command '%s', please run it separately first before using 'from_cmd'", command)
            raise RuntimeError
        
        return temp_dir_name

    def to_csv(self, *, errors='trunc'):
        """"""
        result_df = self._to_df(errors=errors)
        self._save_result(result_df)

    def _to_df(self, *, errors):
        if not any(value.size for value in self._info.values()):
            logger.error("'from_args' or 'from_cmd' should be called first before invoking this method")
            logger.error("No result will be saved")
            raise RuntimeError
        
        # Does not support match expression until Python >= 3.10 
        if errors == 'pad':
            self._padding()
        elif errors == 'trunc':
            self._truncating()
        elif errors == 'strict':
            pass
        else:
            logger.error("Wrong value 'errors'. Expected to be either '%s', '%s' or '%s', got '%s'", 'pad', 'trunc', 'strict', errors)
            raise ValueError
        
        return pd.DataFrame(self._info)
    
    def _padding(self):
        max_len = max(map(len, self._info.values()))

        for key in self._info.keys():
            array = self._info[key]
            diff = max_len - len(array)
            if diff != 0:
                self._info[key] = np.append(array, [np.nan] * diff)
    
    def _truncating(self):
        min_len = min(map(len, self._info.values()))

        for key in self._info.keys():
            array = self._info[key]
            self._info[key] = array[:min_len]
    
    def _save_result(self, df_to_save: pd.DataFrame, suffix='.csv'):
        path = self._get_candidate_path(suffix=suffix)

        flags = os.O_WRONLY | os.O_CREAT
        modes = os.st.S_IRUSR | os.st.S_IRGRP
        with os.fdopen(os.open(path, flags, modes), 'w') as file:
            df_to_save.to_csv(file, encoding='utf-8', index=False)
        
        logger.info("Sythesizer has finished his work, the result is stored at '%s'", path)

    def _get_candidate_path(self, suffix):
        prefix = 'msit_synthesizer_result_'
        return prefix + get_timestamp() + suffix
