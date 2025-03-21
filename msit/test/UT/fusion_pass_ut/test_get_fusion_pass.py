import os
import json
import tempfile
from unittest.mock import patch
import pandas as pd
import numpy as np

from components.debug.compare.fusion_pass_cmp.get_fusion_pass import (get_single_op_info_from_op_list,
                                                                      read_fusion_pass_from_json, write_fusion_to_csv,
                                                                      write_output_to_csv, logger)

MOCK_JSON_DATA = {
    "graph": [
        {
            "op": [
                {
                    "name": "op1",
                    "attr": [
                        {"key": "pass_name", "value": {"list": {"s": ["pass1"]}}},
                        {"key": "pass_name_ub", "value": {"s": "pass3"}}
                    ]
                },
                {
                    "name": "op2",
                    "attr": [
                        {"key": "pass_name_ub", "value": {"s": "pass4"}}
                    ]
                }
            ]
        },
        {
            "op": [
                {
                    "name": "op3",
                    "attr": [
                        {"key": "pass_name", "value": {"list": {"s": ["pass5"]}}}
                    ]
                }
            ]
        }
    ]
}

MOCK_CSV_DATA = """Index,Address.1,DataType.1,CompareFailReason,NPUDump,TensorIndex,CosineSimilarity
1,0x1000,float32,,op1,output,0.99
2,0x2000,float32,,op2,output,NaN
3,0x3000,float32,,op3,input,0.95
4,0x4000,float32,,op4,output,0.98
"""

MOCK_OP_INFO_DICT = {
    "op1": {"pass1"},
    "op2": {"pass3"},
    "op4": {"pass4"}
}


def test_get_single_op_info_from_op_list():
    op_list = [
        {
            "name": "op1",
            "attr": [
                {"key": "pass_name", "value": {"list": {"s": ["pass1"]}}},
                {"key": "pass_name_ub", "value": {"s": "pass3"}}
            ]
        },
        {
            "name": "op2",
            "attr": [
                {"key": "pass_name", "value": {"list": {"s": ["pass4"]}}}
            ]
        },
        {
            "name": "op3",
            "attr": []
        }
    ]
    expected_op_info_dict = {
        "op1": {"pass1", "pass3"},
        "op2": {"pass4"}
    }

    op_info_dict = get_single_op_info_from_op_list(op_list)
    assert op_info_dict == expected_op_info_dict


def create_temp_json_file(data):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(data, f)
        return f.name


def test_read_fusion_pass_from_json_success():
    json_file_path = create_temp_json_file(MOCK_JSON_DATA)
    result = read_fusion_pass_from_json(json_file_path)
    expected_result = {
        "op1": {"pass1", "pass3"},
        "op2": {"pass4"},
        "op3": {"pass5"}
    }
    assert result == expected_result


def test_read_fusion_pass_from_json_file_not_found():
    result = read_fusion_pass_from_json("non_existent_file.json")
    assert result == {}


def test_read_fusion_pass_from_json_invalid_json():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write("invalid json content")
        json_file_path = f.name
    result = read_fusion_pass_from_json(json_file_path)
    assert result == {}


def test_read_fusion_pass_from_json_file_too_large():
    with patch('components.utils.file_open_check.ms_open', side_effect=Exception("File too large")):
        result = read_fusion_pass_from_json("any_file.json")
        assert result == {}


def create_temp_csv_file(data):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(data)
        return f.name


def test_write_fusion_to_csv():
    input_csv = create_temp_csv_file(MOCK_CSV_DATA)
    output_csv = create_temp_csv_file("")  # Create an empty file

    with patch('components.utils.check.rule.Rule.input_file') as mock_rule:
        mock_rule.return_value.check.return_value = True  # Simulate a successful check
        write_fusion_to_csv(input_csv, MOCK_OP_INFO_DICT, output_csv)
    assert os.path.exists(output_csv), "Output CSV file was not created"
    assert os.path.getsize(output_csv) > 0, "Output CSV file is empty"
    
    output_df = pd.read_csv(output_csv)
    expected_data = {
        "NPUDump": ["op1", "op2", "op4"],
        "TensorIndex": ["output", "output", "output"],
        "CosineSimilarity": [0.99, np.nan, 0.98],  # Use np.nan for NaN values
        "PassName": ["{'pass1'}", "{'pass3'}", "{'pass4'}"],
        "MatchError": [np.nan, "Fusion node not match", np.nan]
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["CosineSimilarity"] = expected_df["CosineSimilarity"].astype(float)

    pd.testing.assert_frame_equal(output_df, expected_df)


def test_write_output_to_csv():
    input_csv = create_temp_csv_file(MOCK_CSV_DATA)
    output_csv = create_temp_csv_file("")  # Create an empty file

    with patch('components.utils.check.rule.Rule.input_file') as mock_rule:
        mock_rule.return_value.check.return_value = True  # Simulate a successful check
        write_output_to_csv(input_csv, MOCK_OP_INFO_DICT, output_csv)
    assert os.path.exists(output_csv), "Output CSV file was not created"
    assert os.path.getsize(output_csv) > 0, "Output CSV file is empty"
    
    output_df = pd.read_csv(output_csv)
    expected_data = {
        "NPUDump": ["op1", "op2", "op4"],
        "TensorIndex": ["output", "output", "output"],
        "CosineSimilarity": [0.99, np.nan, 0.98],
        "PassName": ["{'pass1'}", "{'pass3'}", "{'pass4'}"],
        "MatchError": [np.nan, "Node not match", np.nan]
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["CosineSimilarity"] = expected_df["CosineSimilarity"].astype(float)
    
    pd.testing.assert_frame_equal(output_df, expected_df)


def test_write_fusion_to_csv_file_not_found():
    with patch.object(logger, 'error') as mock_logger:
        with patch('components.utils.check.rule.Rule.input_file') as mock_rule:
            mock_rule.return_value.check.side_effect = Exception("File not found")
            write_fusion_to_csv("nonexistent.csv", MOCK_OP_INFO_DICT, "output.csv")
        mock_logger.assert_called_once_with('load input csv failed, err:File not found')


def test_write_output_to_csv_file_not_found():
    with patch.object(logger, 'error') as mock_logger:
        with patch('components.utils.check.rule.Rule.input_file') as mock_rule:
            mock_rule.return_value.check.side_effect = Exception("File not found")
            write_output_to_csv("nonexistent.csv", MOCK_OP_INFO_DICT, "output.csv")
        mock_logger.assert_called_once_with('load input csv failed, err:File not found')
