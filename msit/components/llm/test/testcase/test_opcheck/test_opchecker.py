from unittest.mock import patch
import pytest

from msit_llm.opcheck.opchecker import _is_atb_only_saved_before
from msit_llm.common.log import logger


@pytest.fixture()
def mock_logger():
    with patch.object(logger, 'error') as mock_error:
        yield mock_error


@pytest.mark.parametrize(
    "dump_scenario, expected_result",
    [
        (['before'], True),
        (['after'], False),
        (['before', 'after'], False)
    ]
)
def test_is_atb_only_saved_before(mock_logger, tmpdir, dump_scenario, expected_result):
    input_path = tmpdir.mkdir('token_id')
    layer_dir = input_path.mkdir('0_Decoder_layer')
    for folders in dump_scenario:
        layer_dir.join(folders).ensure()
    res = _is_atb_only_saved_before(str(input_path))

    assert res is expected_result
    mock_logger.assert_not_called()


def test_is_atb_only_saved_before_false_no_folders(mock_logger, tmpdir):
    input_path = tmpdir.mkdir('token_id')
    res = _is_atb_only_saved_before(str(input_path))

    assert res is False
    mock_logger.assert_called_once()
