import argparse
import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import parse_data_to_trace as parse


@patch("os.walk")
def test_find_config_files(mock_walk):
    mock_walk.return_files = [
        ('/fake_path', ['host'], []),
        ('/fake_path/host', [], ['host_start.log', 'info.json']),
    ]
    folder_path = '/fake_path'
    config_path, info_path = parse.find_config_files(folder_path)

    assert config_path == '/fake_path/host/host_start.log'
    assert info_path == '/fake_path/host/info.json'


@patch("os.walk")
def test_find_config_files_when_no_files(mock_walk):
    mock_walk.return_files = [
        ('/fake_path', ['host'], []),
        ('/fake_path/host', [], ['other_files.txt', 'info.json']),
    ]
    folder_path = '/fake_path'

    with pytest.raises(ValueError):
        parse.find_config_files(folder_path)


@pytest.mark.parametrize(
    "message, mark_id, exp_return", [
        ("span=1234|message", 1234, ("1234", "message")),
        ("message_without_span", 1234, ("1234", "message_without_span")),
    ]
)
def test_extract_span_info_from_message(message, mark_id, exp_return):
    span_id, message = parse.extract_span_info_from_message(message, mark_id)
    assert span_id == exp_return[0]
    assert message == exp_return[1]


@pytest.mark.parametrize(
    "message, exp_return", [
        ('{"name": "aa"}', {"name": "aa"}),
        ('"name": "aa",', {"name": "aa"}),
    ]
)
def test_convert_message_to_json(message, exp_return):
    result = parse.convert_message_to_json(message)
    assert result == exp_return


def test_get_state_name_by_value_when_valid():
    assert parse.get_state_name_by_value(0) == "WAITING"
    assert parse.get_state_name_by_value(1) == "PENDING"
    assert parse.get_state_name_by_value(2) == "RUNNING"
    assert parse.get_state_name_by_value(3) == "SWAPPED"
    assert parse.get_state_name_by_value(4) == "RECOMPUTED"
    assert parse.get_state_name_by_value(5) == "SUSPENDED"
    assert parse.get_state_name_by_value(6) == "END"
    assert parse.get_state_name_by_value(7) == "STOP"
    assert parse.get_state_name_by_value(8) == "PREFILL_HOLD"


def test_get_state_name_by_value_when_invalid():
    assert parse.get_state_name_by_value(10) == "10"
    assert parse.get_state_name_by_value("10") == "10"


def test_get_state_name_by_value_when_none():
    with pytest.raises(ValueError):
        parse.get_state_name_by_value(None)


mock_cpu_data = [
    (3, 0, 0.7),
    (2, 1, 0.5),
    (1, 'Avg', 0.6),
]
mock_cpu_columns = ['start_time', 'cpu_no', 'usage']


@patch('sqlite3.connect')
def test_load_cpu_data_from_database(mock_connect):

    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.return_value = mock_cpu_data
    mock_cursor.description = [(col,) for col in mock_cpu_columns]

    db_path = 'fake_path.db'
    cpu_data_df = parse.load_cpu_data_from_database(db_path)

    mock_connect.assert_called_once_with(db_path)
    mock_cursor.excute.assert_called_once_with("""
        SELECT *
        FROM CpuUsage
        WHERE cpu_no == 'Avg'
    """)
    assert isinstance(cpu_data_df, pd.DataFrame)
    assert cpu_data_df.shape == (3, 3)
    assert list(cpu_data_df.columns) == ['start_time', 'cpu_no', 'usage']
    assert cpu_data_df['cpu_no'].iloc[0] == 'Avg'
    assert cpu_data_df['usage'].iloc[0] == '0.6'

    mock_conn.close.assert_called_once()


def mock_load_cpu_data_from_database(db_path):
    return pd.DataFrame(mock_cpu_data, columns=mock_cpu_columns)


@patch('os.walk')
@patch('parse_data_to_trace.load_cpu_data_from_database', side_effect=mock_load_cpu_data_from_database)
def test_find_cpu_data_from_folder(mock_load, mock_walk):
    mock_walk.return_value = [
        ('/folder_path', [], ['host_cpu_usage.db']),
    ]
    folder_path = '/folder_path'
    result = parse.find_cpu_data_from_folder(folder_path)

    assert result.shape(3, 3)
    assert result['start_time'].iloc[0] == 1
    assert result['start_time'].iloc[2] == 3
    mock_load.assert_called_once()


@patch('os.walk')
@patch('parse_data_to_trace.load_cpu_data_from_database', side_effect=mock_load_cpu_data_from_database)
def test_find_cpu_data_from_folder_when_failed(mock_load, mock_walk):
    mock_walk.return_value = [
        ('/folder_path', [], []),
    ]
    folder_path = '/folder_path'
    with pytest.raises(ValueError, match=f"No valid cpu database found in {folder_path}, please check."):
        parse.find_cpu_data_from_folder(folder_path)
        mock_load.assert_called_once()


@patch('sqlite3.connect')
def test_load_data_from_database(mock_connect):
    mock_data = [
        (10000, 1010, "span=1|message"),
        (10000, 1020, "span=2|message"),
    ]
    mock_columns = ['pid', 'tid', 'message']
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.return_value = mock_data
    mock_cursor.description = [(col,) for col in mock_columns]

    db_path = 'fake_path.db'
    data_df = parse.load_data_from_database(db_path)

    mock_connect.assert_called_once_with(db_path)
    mock_cursor.excute.assert_called_once_with("""
        SELECT *
        FROM MsprofTxEx
    """)
    assert isinstance(data_df, pd.DataFrame)
    assert data_df.shape == (2, 3)
    assert list(data_df.columns) == ['pid', 'tid', 'message']
    assert data_df['pid'].iloc[0] == 10000
    assert data_df['tid'].iloc[0] == 1010
    assert data_df['message'].iloc[0] == "span=1|message"

    mock_conn.close.assert_called_once()


@pytest.mark.parametrize(
    "message, exp_return", [
        ({'name': 'httpReq', 'recvTokenSize': 123}, {'recvTokenSize': 123}),
        ({'name': 'ReqEnQueue', 'queue': 5, 'size': 10}, {'queueID': 5, 'queueSize': 10}),
        ({'name': 'deviceKvCache', 'value': 123, 'event': 'KvCache'}, {'KvCacheValue': 123, 'name': 'KvCache'}),
        ({'name': 'hostKvCache', 'value': 123, 'event': 'KvCache'}, {'KvCacheValue': 123, 'name': 'KvCache'}),
        ({'name': 'ReqDeQueue', 'queue': 5, 'size': 10}, {'queueID': 5, 'queueSize': 10}),
        ({'name': 'httpRes', 'replyTokenSize': 123}, {'replyTokenSize': 123}),
        ({'name': 'ReqEnQueue'}, {'queueID': None, 'queueSize': None}),
        ({}, {}),
        ({'name': 'unknown'}, {}),
    ]
)
def test_add_args_for_state_type(message, exp_return):
    result = parse.add_args_for_state_type(message)
    assert result == exp_return


@patch('os.patch.exists')
@patch('os.access')
@patch('os.makedirs')
@patch('os.path.abspath')
def test_check_output_path_valid_when_path_not_exists(mock_abspath, mock_makedirs, mock_access, mock_exists):
    path = "folder_path/output"

    mock_abspath.return_value = path
    mock_exists.return_value = False
    mock_access.return_value = True

    result = parse.check_output_path_valid(path)
    assert result == path
    mock_makedirs.assert_called_once_with(path)


@patch('os.patch.exists')
@patch('os.access')
@patch('os.makedirs')
@patch('os.path.abspath')
def test_check_output_path_valid_when_path_valid(mock_abspath, mock_makedirs, mock_access, mock_exists):
    path = "folder_path/output"

    mock_abspath.return_value = path
    mock_exists.return_value = True
    mock_access.return_value = True

    result = parse.check_output_path_valid(path)
    assert result == path
    mock_makedirs.assert_not_called(path)


@patch('os.patch.exists')
@patch('os.access')
@patch('os.makedirs')
@patch('os.path.abspath')
def test_check_output_path_valid_when_path_not_writable(mock_abspath, mock_makedirs, mock_access, mock_exists):
    path = "folder_path/output"

    mock_abspath.return_value = path
    mock_exists.return_value = True
    mock_access.return_value = False

    with pytest.raises(argparse.ArgumentTypeError):
        parse.check_output_path_valid(path)
    mock_makedirs.assert_not_called(path)


@patch('os.patch.exists')
@patch('os.path.isdir')
def test_check_input_path_valid_when_path_not_exists(mock_isdir, mock_exists):
    path = "folder_path/input"

    mock_exists.return_value = False
    mock_isdir.return_value = True

    with pytest.raises(argparse.ArgumentTypeError, match=f"Path does not exist: {path}"):
        parse.check_input_path_valid(path)


@patch('os.patch.exists')
@patch('os.path.isdir')
def test_check_input_path_valid_when_path_not_dir(mock_isdir, mock_exists):
    path = "folder_path/input"

    mock_exists.return_value = True
    mock_isdir.return_value = False

    with pytest.raises(argparse.ArgumentTypeError, match=f"Path is not a valid directory: {path}"):
        parse.check_input_path_valid(path)


@patch('os.patch.exists')
@patch('os.path.isdir')
def test_check_input_path_valid_when_path_illegal(mock_isdir, mock_exists):
    path = "folder_path/../input"

    mock_exists.return_value = True
    mock_isdir.return_value = False

    with pytest.raises(argparse.ArgumentTypeError, match=f"Path contains illegal characters: {path}"):
        parse.check_input_path_valid(path)


@patch('os.patch.exists')
@patch('os.path.isdir')
def test_check_input_path_valid_when_path_valid(mock_isdir, mock_exists):
    path = "folder_path/input"

    mock_exists.return_value = True
    mock_isdir.return_value = True

    result = parse.check_input_path_valid(path)
    assert result == path
