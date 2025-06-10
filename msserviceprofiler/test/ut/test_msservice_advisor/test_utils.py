import pytest
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import namedtuple
import tempfile
import shutil

from msserviceprofiler.msservice_advisor.profiling_analyze import utils


# Helper functions for file/directory testing
def create_temp_file(size_bytes, content=None):
    """Create a temporary file with specified size"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    if content:
        temp_file.write(content.encode())
    else:
        temp_file.write(b'0' * size_bytes)
    temp_file.close()
    return temp_file.name

def create_temp_dir_with_files(file_count, file_size=1):
    """Create a temporary directory with specified number of files"""
    temp_dir = tempfile.mkdtemp()
    for i in range(file_count):
        with open(os.path.join(temp_dir, f"file_{i}.txt"), 'w') as f:
            f.write('0' * file_size)
    return temp_dir

# Test str_ignore_case
def test_str_ignore_case_given_normal_string_when_lowercase_then_processed():
    assert utils.str_ignore_case("Hello") == "hello"
    assert utils.str_ignore_case("HELLO_WORLD") == "helloworld"
    assert utils.str_ignore_case("Mix-Ed_Case") == "mixedcase"

def test_str_ignore_case_given_empty_string_when_processed_then_empty():
    assert utils.str_ignore_case("") == ""

def test_str_ignore_case_given_non_string_when_processed_then_error():
    with pytest.raises(AttributeError):
        utils.str_ignore_case(None)
    with pytest.raises(AttributeError):
        utils.str_ignore_case(123)

def test_str_ignore_case_given_unicode_string_when_processed_then_lowercase():
    assert utils.str_ignore_case("Héllö_Wörld") == "héllöwörld"

# Test walk_dict
def test_walk_dict_given_flat_dict_when_walked_then_yields_items():
    data = {"a": 1, "b": 2}
    result = list(utils.walk_dict(data))
    assert ("a", 1, "") in result
    assert ("b", 2, "") in result

def test_walk_dict_given_nested_dict_when_walked_then_yields_all_items():
    data = {"a": {"b": 2, "c": {"d": 3}}}
    result = list(utils.walk_dict(data))
    assert ("b", 2, "a") in result
    assert ("d", 3, "a.c") in result

def test_walk_dict_given_mixed_structure_when_walked_then_yields_all_items():
    data = {"a": [1, {"b": 2}, (3, 4)]}
    result = list(utils.walk_dict(data))
    # Note: The list/tuple handling in the original function has a bug (undefined 'key')
    # We'll test the expected behavior assuming it's fixed
    assert (0, 1, "a") in result or ("0", 1, "a") in result
    assert ("b", 2, "a.1") in result

def test_walk_dict_given_empty_structure_when_walked_then_no_yield():
    assert list(utils.walk_dict({})) == []
    assert list(utils.walk_dict([])) == []
    assert list(utils.walk_dict(())) == []

def test_walk_dict_given_non_dict_when_walked_then_no_yield():
    assert list(utils.walk_dict("string")) == []
    assert list(utils.walk_dict(123)) == []

# Test set_log_level
def test_set_log_level_given_valid_level_when_set_then_level_changed():
    original_level = utils.logger.level
    utils.set_log_level("debug")
    assert utils.logger.level == logging.DEBUG
    utils.set_log_level("INFO")
    assert utils.logger.level == logging.INFO
    utils.set_log_level(original_level)  # Restore

def test_set_log_level_given_invalid_level_when_set_then_warning_logged(caplog):
    utils.set_log_level("invalid")
    assert "Set invalid log level failed" in caplog.text

def test_set_log_level_given_case_variations_when_set_then_works():
    utils.set_log_level("Debug")
    assert utils.logger.level == logging.DEBUG
    utils.set_log_level("ERROR")
    assert utils.logger.level == logging.ERROR

# Test set_logger
def test_set_logger_given_new_logger_when_configured_then_has_handler():
    test_logger = utils.logging.getLogger("test_logger")
    utils.set_logger(test_logger)
    assert len(test_logger.handlers) == 1
    assert test_logger.propagate is False

def test_set_logger_given_existing_handler_when_configured_then_no_duplicate():
    test_logger = logging.getLogger("test_logger2")
    test_logger.addHandler(logging.StreamHandler())
    utils.set_logger(test_logger)
    assert len(test_logger.handlers) == 1  # Original handler replaced?

# Test vaild_readable_directory
def test_vaild_readable_directory_given_valid_dir_when_checked_then_passes():
    temp_dir = tempfile.mkdtemp()
    try:
        utils.vaild_readable_directory(temp_dir)  # Should not raise
    finally:
        os.rmdir(temp_dir)

def test_vaild_readable_directory_given_nonexistent_path_when_checked_then_error():
    with pytest.raises(FileExistsError):
        utils.vaild_readable_directory("/nonexistent/path")

def test_vaild_readable_directory_given_file_path_when_checked_then_error():
    temp_file = create_temp_file(10)
    try:
        with pytest.raises(ValueError):
            utils.vaild_readable_directory(temp_file)
    finally:
        os.unlink(temp_file)

def test_vaild_readable_directory_given_unreadable_dir_when_checked_then_error():
    temp_dir = tempfile.mkdtemp()
    try:
        os.chmod(temp_dir, 0o000)  # Remove all permissions
        with pytest.raises(PermissionError):
            utils.vaild_readable_directory(temp_dir)
    finally:
        os.chmod(temp_dir, 0o755)  # Restore permissions for cleanup
        os.rmdir(temp_dir)

# Test vaild_readable_file
def test_vaild_readable_file_given_valid_file_when_checked_then_returns_path():
    temp_file = create_temp_file(10)
    try:
        result = utils.vaild_readable_file(temp_file)
        assert isinstance(result, Path)
    finally:
        os.unlink(temp_file)

def test_vaild_readable_file_given_oversized_file_when_checked_then_error():
    oversized = utils.MAX_FILE_SIZE * utils.BYTES_TO_GB + 1
    temp_file = create_temp_file(oversized)
    try:
        with pytest.raises(ValueError):
            utils.vaild_readable_file(temp_file)
    finally:
        os.unlink(temp_file)

def test_vaild_readable_file_given_unreadable_file_when_checked_then_error():
    temp_file = create_temp_file(10)
    try:
        os.chmod(temp_file, 0o000)  # Remove all permissions
        with pytest.raises(PermissionError):
            utils.vaild_readable_file(temp_file)
    finally:
        os.chmod(temp_file, 0o644)  # Restore permissions for cleanup
        os.unlink(temp_file)

# Test get_directory_size
def test_get_directory_size_given_empty_dir_when_calculated_then_zero():
    temp_dir = tempfile.mkdtemp()
    try:
        size = utils.get_directory_size(temp_dir)
        assert size == 0.0
    finally:
        os.rmdir(temp_dir)

def test_get_directory_size_given_dir_with_files_when_calculated_then_correct():
    temp_dir = create_temp_dir_with_files(3, 1024)  # 3 files of 1KB each
    try:
        size = utils.get_directory_size(temp_dir)
        expected = (3 * 1024) / BYTES_TO_GB
        assert pytest.approx(size) == expected
    finally:
        shutil.rmtree(temp_dir)

def test_get_directory_size_given_excess_iterations_when_calculated_then_error():
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [("/fake", [], ["file"])] * (utils.MAX_FILE_ITER_TIME + 1)
        with pytest.raises(ValueError):
            utils.get_directory_size("/fake/path")

def test_get_directory_size_given_symlinks_when_calculated_then_ignored():
    temp_dir = tempfile.mkdtemp()
    temp_file = create_temp_file(1024)
    try:
        os.symlink(temp_file, os.path.join(temp_dir, "symlink"))
        size = utils.get_directory_size(temp_dir)
        assert size == 0.0  # Symlinks should be ignored
    finally:
        os.unlink(temp_file)
        shutil.rmtree(temp_dir)
