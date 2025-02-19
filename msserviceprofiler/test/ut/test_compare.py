import pytest
import pandas as pd
import sqlite3
from pathlib import Path
import shutil
from unittest.mock import patch
from ms_service_profiler_ext.compare import (
    add_compare_visual_db_table,
    compare_csv,
    compare,
    report,
    visualize,
    main,
    check_input_path_valid,
    check_output_path_valid,
    set_log_level,
    logger
)

# Mocking the logger to avoid errors
from unittest.mock import MagicMock
logger.warning = MagicMock()

# Test cases for add_compare_visual_db_table
def test_add_compare_visual_db_table_given_valid_input_when_write_to_db_then_success():
    # Arrange
    db_filepath = "test.db"
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    table_name = "test_table"

    # Act
    add_compare_visual_db_table(db_filepath, df, table_name)

    # Assert
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    result = cursor.fetchall()
    conn.close()
    assert len(result) == 2

def test_add_compare_visual_db_table_given_invalid_db_path_when_write_to_db_then_raise_error():
    # Arrange
    db_filepath = "/invalid/path/test.db"
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    table_name = "test_table"

    # Act & Assert
    with pytest.raises(ValueError):
        add_compare_visual_db_table(db_filepath, df, table_name)

# Test cases for compare_csv
def test_compare_csv_given_valid_csv_files_when_compare_then_correct_result():
    # Arrange
    fp_a = "file_a.csv"
    fp_b = "file_b.csv"
    df_a = pd.DataFrame({'Metric': ['m1', 'm2'], 'value': [10, 20]})
    df_b = pd.DataFrame({'Metric': ['m1', 'm2'], 'value': [15, 25]})
    df_a.to_csv(fp_a, index=False)
    df_b.to_csv(fp_b, index=False)

    # Act
    result = compare_csv(fp_a, fp_b)

    # Assert
    assert len(result) == 3  # 3 rows: a, b, diff

def test_compare_csv_given_different_columns_when_compare_then_raise_error():
    # Arrange
    fp_a = "file_a.csv"
    fp_b = "file_b.csv"
    df_a = pd.DataFrame({'Metric': ['m1', 'm2'], 'value': [10, 20]})
    df_b = pd.DataFrame({'Metric': ['m1', 'm2'], 'other_value': [15, 25]})
    df_a.to_csv(fp_a, index=False)
    df_b.to_csv(fp_b, index=False)

    # Act & Assert
    with pytest.raises(ValueError):
        compare_csv(fp_a, fp_b)

# Test cases for compare
def test_compare_given_valid_input_dirs_when_compare_then_correct_result():
    # Arrange
    input_a = "dir_a"
    input_b = "dir_b"
    Path(input_a).mkdir(exist_ok=True)
    Path(input_b).mkdir(exist_ok=True)
    (Path(input_a) / "service_summary.csv").touch()
    (Path(input_b) / "service_summary.csv").touch()

    # Act
    result = compare(input_a, input_b)

    # Assert
    assert "service" in result

def test_compare_given_invalid_input_dirs_when_compare_then_empty_result():
    # Arrange
    input_a = "invalid_dir_a"
    input_b = "invalid_dir_b"

    # Act
    result = compare(input_a, input_b)

    # Assert
    assert not result

# Test cases for report
def test_report_given_valid_results_when_write_to_excel_then_success():
    # Arrange
    results = {"service": pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})}
    output_path = "test_output.xlsx"

    # Act
    report(results, output_path)

    # Assert
    df = pd.read_excel(output_path, sheet_name="service")
    assert len(df) == 2

# Test cases for visualize
def test_visualize_given_valid_results_when_write_to_db_then_success():
    # Arrange
    results = {"service": pd.DataFrame({'Metric': ['m1', 'm2'], 'value': [10, 20]})}
    output_path = Path("test_output.db")

    # Act
    visualize(results, output_path)

    # Assert
    assert output_path.exists()

# Test cases for main
@patch('ms_service_profiler_ext.compare.compare')
@patch('ms_service_profiler_ext.compare.report')
@patch('ms_service_profiler_ext.compare.visualize')
def test_main_given_valid_args_when_run_then_success(mock_visualize, mock_report, mock_compare):
    # Arrange
    mock_compare.return_value = {"service": pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})}
    args = ["input_a", "input_b", "--output-path", "output_dir", "--log-level", "info"]

    # Act
    with patch('sys.argv', ['compare.py'] + args):
        main()

    # Assert
    mock_compare.assert_called_once()
    mock_report.assert_called_once()
    mock_visualize.assert_called_once()

# Test cases for check_input_path_valid
def test_check_input_path_valid_given_valid_path_when_check_then_return_path():
    # Arrange
    valid_path = "valid_path"
    Path(valid_path).mkdir(exist_ok=True)

    # Act
    result = check_input_path_valid(valid_path)

    # Assert
    assert result == valid_path

def test_check_input_path_valid_given_invalid_path_when_check_then_raise_error():
    # Arrange
    invalid_path = "invalid_path"

    # Act & Assert
    with pytest.raises(argparse.ArgumentTypeError):
        check_input_path_valid(invalid_path)

# Test cases for check_output_path_valid
def test_check_output_path_valid_given_valid_path_when_check_then_return_path():
    # Arrange
    valid_path = "valid_output_path"
    Path(valid_path).mkdir(exist_ok=True)

    # Act
    result = check_output_path_valid(valid_path)

    # Assert
    assert result == valid_path

def test_check_output_path_valid_given_invalid_path_when_check_then_raise_error():
    # Arrange
    invalid_path = "/invalid/output/path"

    # Act & Assert
    with pytest.raises(argparse.ArgumentTypeError):
        check_output_path_valid(invalid_path)

# Test cases for set_log_level
def test_set_log_level_given_valid_level_when_set_then_success():
    # Arrange
    log_level = "info"

    # Act
    set_log_level(log_level)

    # Assert
    assert logger.level == "INFO"

def test_set_log_level_given_invalid_level_when_set_then_raise_error():
    # Arrange
    log_level = "invalid_level"

    # Act & Assert
    with pytest.raises(ValueError):
        set_log_level(log_level)