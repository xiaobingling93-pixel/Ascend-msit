# test_vllm_command.py
import pytest
from unittest.mock import patch
import shutil
from msserviceprofiler.modelevalstate.config.custom_command import VllmCommand, VllmCommandConfig

# Mock for VllmCommandConfig since it's not provided in the original code
class MockVllmCommandConfig:
    pass

class TestVllmCommand:
    @patch('shutil.which')
    def test_init_success(self, mock_which):
        """Test successful initialization when vllm is found in PATH"""
        # Setup
        mock_which.return_value = "/usr/bin/vllm"
        config = MockVllmCommandConfig()
        
        # Execute
        command = VllmCommand(config)
        
        # Verify
        assert command.process == "/usr/bin/vllm"
        assert command.command_config == config

    @patch('shutil.which')
    def test_init_failure_vllm_not_found(self, mock_which):
        """Test initialization fails when vllm is not found in PATH"""
        # Setup
        mock_which.return_value = None
        config = MockVllmCommandConfig()
        
        # Execute & Verify
        with pytest.raises(ValueError) as excinfo:
            VllmCommand(config)
        
        assert "Error: The 'vllm' executable was not found in the system PATH." in str(excinfo.value)