import os

import pandas as pd

from .utils import get_timestamp
from ..common.log import logger


class Analyzer(object):
    BAD_CASE_FOLDER_NAME = 'msit_bad_case_analyze'
    BAD_CASE_CSV_PREFIX = 'msit_bad_case_result_'

    @classmethod
    def from_csv(cls, *, golden_csv_path: str, test_csv_path: str) -> None:
        """"""
        golden_df = cls._validate_csv_path(golden_csv_path)
        test_df = cls._validate_csv_path(test_csv_path)

        cls._validate_df(golden_df)
        cls._validate_df(test_df)

        cls._compare_golden_with_test(golden_df, test_df)
    
    @classmethod
    def _validate_csv_path(cls, csv_path: str) -> pd.DataFrame:
        """"""
        if not isinstance(csv_path, str) or not csv_path.endswith('.csv'):
            logger.error("Invalid csv path, only path with suffix '.csv' is allowed: '%s'", csv_path)
            raise ValueError

        try:
            file_status = os.stat(csv_path)
        except (PermissionError, FileNotFoundError, OSError) as e:
            logger.error("%s: %s", e.strerror, csv_path)
            raise

        currrent_uid = os.getuid()
        if file_status.st_uid != currrent_uid and currrent_uid != 0:
            logger.error("Inconsistent owner. '%s' should be used only by its owner or superuser", csv_path)
            raise PermissionError
        
        if (os.st.S_IWOTH & file_status.st_mode) == os.st.S_IWOTH:
            logger.error("unsafe csv path, '%s' should not be other writeable", csv_path)
            raise PermissionError

        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception:
            logger.debug("Stack Info:", stack_info=True)
            raise

        return df

    @classmethod
    def _validate_df(cls, df: pd.DataFrame) -> None:
        df.rename(columns={'pass': 'passed'}, inplace=True)
        df.dropna(inplace=True)

        MAIN_COLUMNS = ['queries', 'input_token_ids', 'output_token_ids', 'passed']
        if any(column not in df.columns for column in MAIN_COLUMNS):
            logger.error("Unmatched csv columns, expected to have 'queries', 'input_token_ids', 'output_token_ids', and 'passed'")
            raise KeyError
    
    @classmethod
    def _compare_golden_with_test(cls, golden_df, test_df) -> None:
        """"""
        merged_df = golden_df.merge(test_df, on='queries', how='left', suffixes=('_golden', '_test'), indicator=True)

        not_equal_mask = merged_df['passed_golden'] != merged_df['passed_test']
        bad_case_mask = not_equal_mask & (merged_df['_merge'] == 'both')
        filtered_df = merged_df[bad_case_mask]

        DESIRED_COLUMNS = ['queries']
        for column in ['input_token_ids', 'output_token_ids', 'passed']:
            DESIRED_COLUMNS.append(f'{column}_golden')
            DESIRED_COLUMNS.append(f'{column}_test')

        cls._save_result(filtered_df[DESIRED_COLUMNS])

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
        os.makedirs(cls.BAD_CASE_FOLDER_NAME, mode=0o700, exist_ok=True)
        path = os.path.join(cls.BAD_CASE_FOLDER_NAME, cls._get_candidate_path(suffix=suffix))

        flags = os.O_WRONLY | os.O_CREAT
        modes = os.st.S_IRUSR | os.st.S_IRGRP
        with os.fdopen(os.open(path, flags, modes), 'w') as file: 
            df_to_save.to_csv(file, encoding='utf-8', index=False)

        logger.info("Analyzer has successfully finished the analysis, the result is stored under '%s'", path)

    @classmethod
    def _get_candidate_path(cls, suffix):
        return cls.BAD_CASE_CSV_PREFIX + get_timestamp() + suffix

    @classmethod
    def from_mixed(cls, *, golden, test) -> None:
        """"""
        from . import Synthesizer
        
        if isinstance(golden, Synthesizer):
            golden = golden._to_df(errors='trunc')
        else:
            golden = cls._validate_csv_path(golden)
        
        if isinstance(test, Synthesizer):
            test = test._to_df(errors='trunc')
        else:
            test = cls._validate_csv_path(test)

        cls._validate_df(golden)
        cls._validate_df(test)

        cls._compare_golden_with_test(golden, test)
