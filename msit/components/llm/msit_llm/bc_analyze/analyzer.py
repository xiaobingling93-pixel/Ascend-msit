import os

import pandas as pd

from msit_llm.bc_analyze.utils import get_timestamp
from msit_llm.common.constant import MSIT_BAD_CASE_FOLDER_NAME
from msit_llm.common.log import logger


class Analyzer(object):
    """This class is used for analyzing the bad case between `golden` and `test` results. `golden` or `test` result
    can be a csv path or a command. If it is a command, `Analyzer` will internally call `Synthesizer.from_cmd` to 
    collect the result for you, and analyze it immediately.
    """
    ANALYZER_FOLDER_NAME = os.path.join(MSIT_BAD_CASE_FOLDER_NAME, 'analyzer')
    BAD_CASE_CSV_PREFIX = 'msit_bad_case_result_'

    @classmethod
    def from_csv(cls, golden_csv_path: str, test_csv_path: str) -> None:
        """Analyze the bad case by comparing `golden_csv_path` with `test_csv_path`, the analysis will be stored 
        as a csv file. If there is no difference between these two, then no report will be saved.
        
        Paramters
        ---------
        `golden_csv_path` : str
            The csv path that is considered to be the golden standard.
        `test_csv_path` : str
            The csv path that is considered to be the test result.

        Notes
        -----
        The analysis csv file will be stored under the directory `msit_bad_case_analyze`, the csv file name will be 
        `msit_bad_case_result_` plus the time stamp. For example, `msit_bad_case_result_20240720042235.csv`.

        Exceptions
        ----------
        `ValueError` : raises if any path does not end with `.csv`
        `OSError` : raises if any path does not exist, no permission to access, or file name too long
        `PermssionError` : raises if the file owner is not the current user, or it is other writeable
        `KeyError` : raises if the header of csv does not include any of 'queries', 'input_token_ids', 
                    'output_token_ids', or 'passed'

        Other errors may be occured by `pandas`

        Examples
        --------
        >>> from msit_llm import Analyzer
        >>> Analyzer.from_csv(csv1, csv2)
        2024-08-07 04:44:08,278 - msit_llm_logger - INFO - 'Analyzer' received two csv paths, the golden one is:
            'csv1'
        and the test one is:
            'csv2'
        2024-08-07 04:45:12,512 - msit_llm_logger - INFO - Checking if path 'csv1' is valid...
        2024-08-07 04:45:13,123 - msit_llm_logger - INFO - Checking if the header of csv is valid...
        2024-08-07 04:45:14,546 - msit_llm_logger - INFO - Checking if path 'csv2' is valid...
        2024-08-07 04:45:15,523 - msit_llm_logger - INFO - Checking if the header of csv is valid...
        2024-08-07 04:45:16,166 - msit_llm_logger - INFO - Analyzing...
        2024-08-07 04:45:17,125 - msit_llm_logger - INFO - 'Analyzer' has successfully finished the analysis ...
        """
        logger.info(
            "'Analyzer' received two csv paths, the golden one is:\n\t'%s'\nand the test one is:\n\t'%s'", 
            golden_csv_path, 
            test_csv_path
        )

        golden_df = cls._validate_csv_path(golden_csv_path)
        test_df = cls._validate_csv_path(test_csv_path)

        cls._validate_df(golden_df)
        cls._validate_df(test_df)

        cls._compare_golden_with_test(golden_df, test_df)
    
    @classmethod
    def _validate_csv_path(cls, csv_path: str) -> pd.DataFrame:
        logger.info("Checking if path '%s' is valid...", csv_path)

        if not isinstance(csv_path, str) or not csv_path.endswith('.csv'):
            logger.error("Invalid csv path, only path with suffix '.csv' is allowed: '%s'", csv_path)
            raise ValueError

        try:
            file_status = os.stat(csv_path)
        except OSError as e:
            logger.error("%s: %s", e.strerror, csv_path)
            raise

        currrent_uid = os.getuid()
        if file_status.st_uid != currrent_uid and currrent_uid != 0:
            logger.error("Inconsistent owner. '%s' should be used only by its owner or superuser", csv_path)
            raise PermissionError
        
        if (os.st.S_IWOTH & file_status.st_mode) == os.st.S_IWOTH:
            logger.error("Unsafe csv path, '%s' should not be other writeable", csv_path)
            raise PermissionError

        return pd.read_csv(csv_path, encoding='utf-8')

    @classmethod
    def _validate_df(cls, df: pd.DataFrame) -> None:
        logger.info("Checking if the header of csv is valid...")
        df.rename(columns={'pass': 'passed'}, inplace=True)
        df.dropna(inplace=True)

        main_columns = ['queries', 'input_token_ids', 'output_token_ids', 'passed']
        if any(column not in df.columns for column in main_columns):
            logger.error(
                "Unmatched csv columns, expected to have 'queries', 'input_token_ids', 'output_token_ids', and 'passed'"
            )
            raise KeyError
    
    @classmethod
    def _compare_golden_with_test(cls, golden_df, test_df) -> None:
        logger.info("Analyzing...")
        merged_df = golden_df.merge(test_df, on='queries', how='left', suffixes=('_golden', '_test'), indicator=True)

        not_equal_mask = merged_df['passed_golden'] != merged_df['passed_test']
        bad_case_mask = not_equal_mask & (merged_df['_merge'] == 'both')
        filtered_df = merged_df[bad_case_mask]

        desired_columns = ['queries']
        for column in ['input_token_ids', 'output_token_ids', 'passed']:
            desired_columns.append(f'{column}_golden')
            desired_columns.append(f'{column}_test')

        cls._save_result(filtered_df[desired_columns])

        not_both_mask = merged_df['_merge'] != 'both'
        not_both_count = not_both_mask.sum()
        if not_both_count != 0:
            unmatched_queries = merged_df['queries'][not_both_mask].head()
            logger.warning(
                "There are '%s' quer(ies) not matched, below is a partial display of these unmatched queries:\n\t '%s'", 
                not_both_count,
                unmatched_queries.to_string(header=False)
            )
        
    @classmethod
    def _save_result(cls, df_to_save: pd.DataFrame, suffix='.csv') -> None:
        if df_to_save.empty:
            logger.warning(
                "'Analyzer' detected that there is no difference between the given golden result and test result. "
                "Hence no result is saved")
            return
        
        os.makedirs(cls.ANALYZER_FOLDER_NAME, mode=0o700, exist_ok=True)
        path = os.path.join(cls.ANALYZER_FOLDER_NAME, cls._get_candidate_path(suffix=suffix))

        flags = os.O_WRONLY | os.O_CREAT
        modes = os.st.S_IRUSR | os.st.S_IWUSR | os.st.S_IRGRP
        with os.fdopen(os.open(path, flags, modes), 'w') as file: 
            df_to_save.to_csv(file, encoding='utf-8', index=False)

        logger.info("'Analyzer' has successfully finished the analysis, the result is stored at '%s'", path)

    @classmethod
    def _get_candidate_path(cls, suffix):
        return cls.BAD_CASE_CSV_PREFIX + get_timestamp() + suffix

    @classmethod
    def from_mixed(cls, golden, test) -> None:
        """This method is designed in case users collect the evaluation result in memory. Asides from path like,
        both `golden` and `test` can be an instance of `Synthesizer`.

        Parameter
        ---------
        `golden` : str or Synthesizer
            The csv path or Sythesizer instance that is considered to be the golden standard.
        `test` : str
            The csv path or Sythesizer instance that is considered to be the test result.

        All the notes, exceptions are consistent to `from_csv`

        Examples
        --------
        >>> from msit_llm import Synthezier, Analyzer
        >>> golden_synthesizer = Synthesizer(
        ...     queries='Question 1', 
        ...     input_token_ids=[1, 2, 3], 
        ...     output_token_ids=[4, 5, 6], 
        ...     passed='Correct')
        >>> Analyzer.from_mixed(golden_synthesizer, test_csv_path)
        2024-08-07 04:45:13,123 - msit_llm_logger - INFO - Checking if the header of csv is valid...
        2024-08-07 04:45:14,546 - msit_llm_logger - INFO - Checking if path 'test_csv_path' is valid...
        2024-08-07 04:45:15,523 - msit_llm_logger - INFO - Checking if the header of csv is valid...
        2024-08-07 04:45:16,166 - msit_llm_logger - INFO - Analyzing...
        2024-08-07 04:45:13,651 - msit_llm_logger - INFO - 'Analyzer' has successfully finished the analysis,
        the result is stored at 'msit_bad_case_analyze/msit_bad_case_result_ieqwe2q5_20240720042235.csv'
        """
        from msit_llm.bc_analyze.synthezier import Synthesizer
        
        if isinstance(golden, Synthesizer):
            golden = golden.to_df(errors='trunc')
        else:
            golden = cls._validate_csv_path(golden)
        
        if isinstance(test, Synthesizer):
            test = test.to_df(errors='trunc')
        else:
            test = cls._validate_csv_path(test)

        cls._validate_df(golden)
        cls._validate_df(test)

        cls._compare_golden_with_test(golden, test)
