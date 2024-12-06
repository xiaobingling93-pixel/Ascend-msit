import pytest
from msit_llm.opcheck.check_case import (
    OpcheckGatingOperation, OpcheckIndexAddOperation, OpcheckNonzeroOperation, OpcheckOnehotOperation,
    OpcheckActivationOperation, OpcheckAllGatherOperation, OpcheckAllReduceOperation, OpcheckBroadcastOperation,
    OpcheckConcatOperation, OpcheckCumsumOperation, OpcheckElewiseAddOperation, OpcheckFillOperation,
    OpcheckGatherOperation, OpcheckKvCacheOperation, OpcheckLinearOperation, OpcheckLinearSparseOperation,
    OpcheckPadOperation, OpcheckPagedAttentionAttentionOperation, OpcheckRepeatOperation,
    OpcheckReshapeAndCacheOperation, OpcheckRmsNormOperation, OpcheckUnpadRopeOperation,
    OpcheckUnpadSelfAttentionOperation, OpcheckSetValueOperation, OpcheckSliceOperation,
    OpcheckSoftmaxOperation, OpcheckSortOperation, OpcheckAddOperation, OpcheckToppOperation,
    OpcheckTransposeOperation, OpcheckUnpadOperation, OpcheckAsStridedOperation, OpcheckLayerNormOperation,
    OpcheckLinearParallelOperation, OpcheckMultinomialOperation, OpcheckReduceOperation,
    OpcheckTransdataOperation, OpcheckWhereOperation, OpcheckMatmulOperation, OpcheckFastSoftMaxOperation,
    OpcheckFastSoftMaxGradOperation, OpcheckElewiseSubOperation, OpcheckRopeGradOperation,
    OpcheckStridedBatchMatmulOperation
)
from msit_llm.opcheck.check_case import OP_NAME_DICT, OutTensorType


# 测试 OP_NAME_DICT 中的每个键值对
@pytest.mark.parametrize("op_name, expected_class", OP_NAME_DICT.items())
def test_op_name_dict_given_valid_op_name_when_valid_then_pass(op_name, expected_class):
    assert OP_NAME_DICT[op_name] == expected_class


# 测试 OutTensorType 枚举中的每个值
@pytest.mark.parametrize("tensor_type, expected_value", [
    (OutTensorType.ACL_DT_UNDEFINED, -1),
    (OutTensorType.ACL_FLOAT, 0),
    (OutTensorType.ACL_FLOAT16, 1),
    (OutTensorType.ACL_INT8, 2),
    (OutTensorType.ACL_INT32, 3),
    (OutTensorType.ACL_UINT8, 4),
    (OutTensorType.ACL_INT16, 6),
    (OutTensorType.ACL_UINT16, 7),
    (OutTensorType.ACL_UINT32, 8),
    (OutTensorType.ACL_INT64, 9),
    (OutTensorType.ACL_UINT64, 10),
    (OutTensorType.ACL_DOUBLE, 11),
    (OutTensorType.ACL_BOOL, 12),
    (OutTensorType.ACL_STRING, 13),
    (OutTensorType.ACL_COMPLEX64, 16),
    (OutTensorType.ACL_COMPLEX128, 17),
    (OutTensorType.ACL_BF16, 27),
    (OutTensorType.ACL_INT4, 29),
    (OutTensorType.ACL_UINT1, 30),
    (OutTensorType.ACL_COMPLEX32, 33)
])
def test_out_tensor_type_given_valid_tensor_type_when_valid_then_pass(tensor_type, expected_value):
    assert tensor_type.value == expected_value


# 测试 OutTensorType 枚举中的每个值是否唯一
def test_out_tensor_type_values_are_unique():
    values = [member.value for member in OutTensorType]
    assert len(values) == len(set(values))


# 测试 OP_NAME_DICT 中的每个类是否可以实例化
@pytest.mark.parametrize("op_name, op_class", OP_NAME_DICT.items())
def test_op_name_dict_classes_can_be_instantiated(op_name, op_class):
    instance = op_class()
    assert isinstance(instance, op_class)
