# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

try:
    from torch.tensor import Tensor
except ModuleNotFoundError as ee:
    from torch import Tensor
import pytest

from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from msmodelslim.common.knowledge_distill.knowledge_distill import get_distill_model
from test.resources.sample_net_torch import TorchTeacherModel
from test.resources.sample_net_torch import TorchStudentModel


@pytest.fixture()
def config():
    distill_config = KnowledgeDistillConfig()
    distill_config.set_hard_label(0.5, 0) \
        .add_inter_soft_label({
            "t_module": "teacher_fc",
            "s_module": "student_fc",
            "t_output_idx": 0,
            "s_output_idx": 0,
            "loss_func": [{"func_name": "KDCrossEntropy",
                           "func_weight": 1}],
        }).add_output_soft_label({
            "t_output_idx": 0,
            "s_output_idx": 0,
            "loss_func": [{"func_name": "KDCrossEntropy",
                           "func_weight": 1}]
        })
    yield distill_config


@pytest.fixture()
def student_model():
    yield TorchStudentModel()


@pytest.fixture()
def teacher_model():
    yield TorchTeacherModel()


class TestDistillModel(object):
    def test_get_distill_model_given_valid_when_any_then_pass(self, config, teacher_model, student_model):
        distill_model = get_distill_model(teacher_model, student_model, config)
        assert distill_model.student_model is student_model
        assert distill_model.teacher_model is teacher_model

    def test_get_student_model_given_valid_when_any_then_pass(self, config, teacher_model, student_model):
        distill_model = get_distill_model(teacher_model, student_model, config)
        input_data = Tensor([[1.0]])
        distill_model(input_data)
        distilled_student = distill_model.get_student_model()
        assert distilled_student is student_model
