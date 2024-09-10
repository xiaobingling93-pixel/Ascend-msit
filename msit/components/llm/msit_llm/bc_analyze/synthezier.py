import os

import numpy as np
import pandas as pd

from msit_llm.bc_analyze.utils import get_timestamp, RandomNameSequence
from msit_llm.common.log import logger
from msit_llm.common.constant import MSIT_BAD_CASE_FOLDER_NAME


class Synthesizer(object):
    """This class is used for collecting llm evluation results under several datasets"""
    HEADER = ['queries', 'input_token_ids', 'output_token_ids', 'passed']
    SYNTHESIZER_FOLDER_NAME = os.path.join(MSIT_BAD_CASE_FOLDER_NAME, 'synthesizer')
    SYNTHESIZER_PREFIX = 'msit_synthesizer_result_'

    _namer = RandomNameSequence()
    
    def __init__(self, *, queries=None, input_token_ids=None, output_token_ids=None, passed=None) -> None:
        """Create a synthesizer collecting information from Large Language Model under dataset evluation
        
        It will internally call `Synthesizer.from_args` for you, for the introduction to the parameters,
        please refer to `Synthesizer.from_args`

        Examples
        --------
        >>> from msit_llm import Synthesizer
        >>> synthesizer = Synthesizer(
        ...     queries='Question 1', 
        ...     input_token_ids=[1, 2, 3], 
        ...     output_token_ids=[4, 5, 6], 
        ...     passed='Correct')
        """
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
    
    def from_args(self, *, queries=None, input_token_ids=None, output_token_ids=None, passed=None) -> None:
        """Collecting dataset evaluation result from memory. User should take care of the evaluation details.
        
        If any parameter is `None`, it will not be added to the `Sythesizer`. This allows the user to add each
        item iteratively whenever it is ready, without extra effort on collecting each item as a whole.

        Parameters:
        -----------
        queries: str or iterable of str, optional
            The queries or questions from the dataset that the model is evaluated on. Default to `None`.
        input_token_ids: Any, optional
            The input ids of the queries or questions after encoding. Default to `None`.
        output_token_ids: Any, optional
            The output ids of the model outputs, the generated texts before decoding. Default to `None`.
        passed: {'Correct', 'Wrong'}, optional
            Indicate whether the generated texts from the model is expected with respect to the dataset label.
            `Correct` or `Wrong` depends on different evluation strategies under corresponding dataset.
            Default to `None`

        Examples
        --------
        >>> from msit_llm import Synthesizer
        >>> synthesizer = Synthesizer()
        >>> synthesizer.from_args(
        ...     queries='Question 1', 
        ...     input_token_ids=[1, 2, 3], 
        ...     output_token_ids=[4, 5, 6], 
        ...     passed='Correct')
        """
        self._update_attributes(
            queries=queries,
            input_token_ids=input_token_ids,
            output_token_ids=output_token_ids,
            passed=passed
        )

    @classmethod
    def from_cmd(cls, command) -> str:
        """Collecting dataset evaluation result from `command`. This method is a static method that will run the 
        `command` in the subprocess and collect the desired results during running, the collecting result will be 
        a csv file that stored at a temporary directory which will be returned by this method.

        Currently only supports `Model Test` from `ATB Speed`.

        Parameters
        ----------
        `command` : str
            The model inference command should be run. It must somehow invoke python interpreter, otherwise 
            nothing will be collected. Currently only supports `Model Test` from `ATB Speed`

        Returns
        -------
        `temp_dir_name` : str
            A temporary directory path name that points to the place where the collecting result csv file is 
            stored.

        Exceptions
        ----------
        `FileNotFoundError` : raise if `patcher` directory is not found in the msit site-packages directory
        `RuntimeError` : raise if an error occured during `command` execution in the subprocess

        Examples
        --------
        The logger prefixes, stdouts and stderrs are omitted in the following examples:
        >>> from msit_llm import Synthesizer
        >>> Synthesizer.from_cmd('bash run.sh pa_fp16 full_CEval 1 1 chatglm2_6b /data/chatglm2_6b 1')
        INFO - Command 'bash run.sh pa_fp16 full_CEval 1 1 chatglm2_6b /data/chatglm2_6b 1' is running...
        ......
        INFO - Command 'bash run.sh pa_fp16 full_CEval 1 1 chatglm2_6b /data/chatglm2_6b 1' returns successfully
        
        An inccorect command:
        >>> Synthesizer.from_cmd('b run.sh')
        ERROR - 'b run.sh': command not found, please run it separately first before using 'from_cmd'

        An error occured in the subprocess:
        >>> Synthesizer.from_cmd('bash run.sh')
        INFO - Command 'bash run.sh' is running...
        ......
        ERROR - Failed to run command 'bash run.sh', please run it separately first before using 'from_cmd'
        """
        import subprocess
        import shlex

        env = os.environ.copy()
        patcher_folder = os.path.join(os.path.dirname(__file__), 'patcher')
        if not os.path.exists(patcher_folder):
            logger.error("Directory '%s' not found, please try to reinstall the latest msit", patcher_folder)
            raise FileNotFoundError

        env['PYTHONPATH'] = patcher_folder + ':' + env.get('PYTHONPATH', '')

        temp_dir_name = 'msit_bad_case_rt' + next(cls._namer)
        env['MSIT_TEMP_DIR_NAME'] = temp_dir_name

        split_command = shlex.split(command)
        if not split_command:
            logger.error("Command is empty,please try to run it first")
            raise ValueError()


        if not split_command:
            logger.error("Command is empty,please try to run it first")
            raise ValueError()

        known_invalid_command = {
            "rm", "mv", "mkfs", "dd",
            "chown", "chmod",
            "shutdown", "reboot",
            "curl", "wget",
        }

        if split_command[0] in known_invalid_command:
            logger.error("Invalid command '%s'", split_command)
            raise ValueError

        logger.debug("Split command %s", split_command)

        try:
            child_process = subprocess.Popen(
                split_command,
                env=env,
            )
        except OSError:
            logger.error(
                "'%s': command not found, please run it separately first before using 'from_cmd'", command
            )
            raise

        logger.info("Command '%s' is running...", command)

        if child_process.wait() != 0:
            logger.error(
                "Failed to run command '%s', please run it separately first before using 'from_cmd'", command
            )
            raise RuntimeError
        
        logger.info("Command '%s' returns successfully", command)
        return temp_dir_name

    def to_csv(self, *, errors='trunc'):
        """Archive the collected result to csv file. 
        
        Users should call `from_args` or implicitly call `from_args` through constructor before invoking this 
        method.

        Parameters
        ----------
        `errors` : {'trunc', 'pad', 'strict'}, optional
            Consistent strategies to the data frame. Since each item can be added to `Synthesizer` asynchronously,
            it is possible to have item with differnt sizes or lengths. If `trunc` is chosen, then the data frame 
            will be truncated to the shortest length among those items. If `pad` is chosen, `Synthesizer` will pad
            missing rows with `np.nan` to be consistent with the largest length. If `strict` is chosen, 
            `Synthesizer` will do nothing and let `pandas` to figure out all the items are consistent or not.

        Notes
        -----
        The resulting csv file name will be `msit_synthesizer_result_` plus a random eight characters string plus 
        time stamp. For example, `msit_synthesizer_result_iew92iq5_20240720042235.csv`.
            
        Exceptions
        ----------
        `RuntimeError` : raises if user did not initalize (or implicitly) `Synthesizer` through `from_args`.
        `ValueError` : raises if unexpected value passed in `errors`, or if the `strict` is chosen, but the 
        `pandas` complains that the data frame is not consistent.

        Examples
        --------
        >>> from msit_llm import Synthesizer
        >>> synthesizer = Synthesizer()
        >>> synthesizer.to_csv()
        ERROR - 'from_args' should be called first before invoking this method
        """
        result_df = self.to_df(errors=errors)
        self._save_result(result_df)

    def to_df(self, *, errors):
        if not any(value.size for value in self._info.values()):
            logger.error("'from_args' should be called first before invoking this method")
            raise RuntimeError
        
        # Does not support match expression until Python >= 3.10 
        if errors == 'pad':
            self._padding()
        elif errors == 'trunc':
            self._truncating()
        elif errors == 'strict':
            pass
        else:
            logger.error(
                "Wrong value 'errors'. "
                "Expected to be either '%s', '%s' or '%s', got '%s'", 'pad', 'trunc', 'strict', errors)
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
        if df_to_save.empty:
            logger.warning("'Synthesizer' detected that there is no data to save. Hence no result is saved")
            return
        
        os.makedirs(self.SYNTHESIZER_FOLDER_NAME, mode=0o700, exist_ok=True)
        path = os.path.join(self.SYNTHESIZER_FOLDER_NAME, self._get_candidate_path(suffix=suffix))

        flags = os.O_WRONLY | os.O_CREAT
        modes = os.st.S_IRUSR | os.st.S_IWUSR | os.st.S_IRGRP
        with os.fdopen(os.open(path, flags, modes), 'w') as file:
            df_to_save.to_csv(file, encoding='utf-8', index=False)
        
        logger.info("'Sythesizer' has successfully finished the synthesis, the result is stored at '%s'", path)

    def _get_candidate_path(self, suffix):
        return self.SYNTHESIZER_PREFIX + next(self._namer) + get_timestamp() + suffix
