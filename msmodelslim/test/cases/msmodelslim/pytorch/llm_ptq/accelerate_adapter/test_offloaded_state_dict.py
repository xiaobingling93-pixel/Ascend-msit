import pytest

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.offloaded_state_dict import DiskStateDictConfig


class TestDiskStateDictConfig:
    @staticmethod
    @pytest.fixture()
    def disk_state_dict_config():
        return DiskStateDictConfig()

    @staticmethod
    def test_return_args(disk_state_dict_config):
        assert disk_state_dict_config.args['save_folder'] is None

    @staticmethod
    def test_set_save_folder_success_when_input_str(disk_state_dict_config):
        disk_state_dict_config.save_folder("path to save")
        assert disk_state_dict_config.args['save_folder'] == "path to save"

    @staticmethod
    def test_set_save_folder_fail_when_input_num(disk_state_dict_config):
        with pytest.raises(ValueError):
            disk_state_dict_config.save_folder(123)

    @staticmethod
    def test_set_save_folder_fail_when_input_None(disk_state_dict_config):
        with pytest.raises(ValueError):
            disk_state_dict_config.save_folder(None)
