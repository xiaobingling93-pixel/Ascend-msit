# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
import tempfile
import json
import argparse

from msmodelslim.pytorch.llm_ptq.mix_calibration.calib_select import CalibrationData, DatasetProcessorBase
from resources.fake_llama.fake_llama import get_fake_llama_model_and_tokenizer

class TestCalibrationData:
    def setup_class(self):
        self.sample_size = {"boolq": 2}
        self.model, self.tokenizer = get_fake_llama_model_and_tokenizer()

        # create a temporary dataset
        with tempfile.NamedTemporaryFile(mode='w+t', suffix='.json', delete=False) as temp_dataset:
            calib_data = [
                {"question": "a", "title": "b", "answer": False, "passage": "d"},
                {"question": "1", "title": "2", "answer": True, "passage": "e"}
            ]
            json.dump(calib_data, temp_dataset)
            temp_dataset.seek(0)
            self.calib_dataset = temp_dataset

        # create a temporary test_config.json
        with tempfile.NamedTemporaryFile(mode='w+t', suffix='.json', delete=False) as temp_config:
            data = {"configurations":
                    [
                        {
                          "dataset_name": "boolq",
                          "dataset_path": f"{self.calib_dataset.name}"
                        }
                    ]
                }
            json.dump(data, temp_config)
            temp_config.seek(0)
            self.config_json = temp_config

    def teardown_class(self):
        self.config_json.close()
        self.calib_dataset.close()

    def test_calibration_data_init(self):
        # Test initialization with default parameters
        calibration = CalibrationData(config_path=self.config_json.name)
        assert len(calibration.handlers) == 1
        calibration = CalibrationData(config_path=self.config_json.name, model=self.model, tokenizer=self.tokenizer)
        assert len(calibration.handlers) == 1
        calibration = CalibrationData(config_path=self.config_json.name, save_path="./test_output.json", model=self.model, tokenizer=self.tokenizer)
        assert len(calibration.handlers) == 1

        # Test initialization with model, but without tokenizer
        with pytest.raises(TypeError):
            CalibrationData(config_path=self.config_json.name, model=self.model)

        # Test initialization with specific model
        with pytest.raises(TypeError):
            CalibrationData(config_path=self.config_json.name, model="not a model")

        # Test initialization with specific config_path
        with pytest.raises(ValueError):
            CalibrationData(config_path="not a path")

        # Test initialization with specific save_path
        with pytest.raises(ValueError):
            CalibrationData(config_path=self.config_json.name, save_path="not a path")

    def test_set_sample_size(self):
        calibration = CalibrationData(config_path=self.config_json.name, model=self.model, tokenizer=self.tokenizer)
        calibration.set_sample_size(self.sample_size)
        assert calibration.handlers["boolq"].sample_size == self.sample_size["boolq"]

        # Test set sample size with wrong type
        with pytest.raises(TypeError):
            fake_sample_size = [2]
            calibration.set_sample_size(fake_sample_size)
        with pytest.raises(TypeError):
            fake_sample_size = {"boolq": "str"}
            calibration.set_sample_size(fake_sample_size)

    def test_set_batch_size(self):
        calibration = CalibrationData(config_path=self.config_json.name, model=self.model, tokenizer=self.tokenizer)

        # Test set batch size with wrong type
        with pytest.raises(argparse.ArgumentTypeError):
            batch_size = "str"
            calibration.set_batch_size(batch_size)

    def test_set_shuffle_seed(self):
        calibration = CalibrationData(config_path=self.config_json.name, model=self.model, tokenizer=self.tokenizer)

        # Test set shuffle seed with wrong type
        with pytest.raises(argparse.ArgumentTypeError):
            seed = "str"
            calibration.set_shuffle_seed(seed)

    def test_add_custormized_dataset_processor(self):
        class CustomizedProcessor(DatasetProcessorBase):
            def __init__(self, dataset_path, tokenizer=None, model=None):
                super().__init__(dataset_path, tokenizer, model)

            def process_data(self, indexs):
                pass

            def verify_positive_prompt(self, prompts, labels):
                pass

        customized_processor = CustomizedProcessor(self.calib_dataset.name)
        calibration = CalibrationData(config_path=self.config_json.name, model=self.model, tokenizer=self.tokenizer)
        calibration.add_custormized_dataset_processor(dataset_name="customized_dataset", processor=customized_processor)
        assert len(calibration.handlers) == 2

        # Test with wrong type
        with pytest.raises(argparse.ArgumentTypeError):
            calibration.add_custormized_dataset_processor(dataset_name=1, processor=customized_processor)
        with pytest.raises(argparse.ArgumentTypeError):
            calibration.add_custormized_dataset_processor(dataset_name="customized", processor=1)