import pytest
import torch
from msit_llm.opcheck.check_case.rms_norm import OpcheckRmsNormOperation, RmsNormType, QuantType, DynamicQuantType

from mock_operation_test import MockOperationTest


OpcheckRmsNormOperation.__bases__ = (MockOperationTest,)

def test_validate_rmsnorm_param_given_valid_layer_type_when_validate_then_correct_result():
    # Arrange
    op = OpcheckRmsNormOperation()
    op.op_param = {'layerType': RmsNormType.RMS_NORM_NORM.value, 'normParam': {'quantType': QuantType.QUANT_INT8.value}}

    # Act
    result = op.validate_rmsnorm_param(RmsNormType.RMS_NORM_NORM.value)

    # Assert
    assert result['quantType'] == QuantType.QUANT_INT8.value
