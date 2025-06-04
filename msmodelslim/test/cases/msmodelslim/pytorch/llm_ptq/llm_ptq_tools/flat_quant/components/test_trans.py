# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.trans import (
    SingleTransMatrix,
    SVDSingleTransMatrix,
    InvSingleTransMatrix,
    DiagonalTransMatrix,
    GeneralMatrixTrans
)


class TestSVDSingleTransMatrix:
    def test_init_default_values(self):
        matrix = SVDSingleTransMatrix(size=10)
        
        assert matrix.size == 10
        assert matrix.deriction == "right"
        assert hasattr(matrix, 'linear_u')
        assert hasattr(matrix, 'linear_v')
        assert hasattr(matrix, 'linear_diag')
        assert matrix._diag_relu is None

    def test_init_with_custom_direction(self):
        matrix = SVDSingleTransMatrix(size=10, deriction="left")
        
        assert matrix.deriction == "left"

    def test_init_with_diag_relu_enabled(self):
        matrix = SVDSingleTransMatrix(size=10, diag_relu=True)
        
        assert matrix._diag_relu is not None
        assert isinstance(matrix._diag_relu, nn.Softplus)

    def test_get_diag_without_relu(self):
        matrix = SVDSingleTransMatrix(size=10, diag_relu=False)
        
        diag = matrix.get_diag()
        
        assert diag.shape == (10,)

    def test_get_diag_with_relu(self):
        matrix = SVDSingleTransMatrix(size=10, diag_relu=True)
        
        diag = matrix.get_diag()
        
        assert diag.shape == (10,)
        assert torch.all(diag > 0)  # Softplus ensures positive values

    def test_get_matrix_not_eval_mode(self):
        matrix = SVDSingleTransMatrix(size=10)
        
        result = matrix.get_matrix()
        
        assert result.shape == (10, 10)

    def test_get_matrix_with_inv_t(self):
        matrix = SVDSingleTransMatrix(size=10)
        
        result = matrix.get_matrix(inv_t=True)
        
        assert result.shape == (10, 10)

    def test_get_matrix_eval_mode(self):
        matrix = SVDSingleTransMatrix(size=10)
        matrix.to_eval_mode()
        
        result = matrix.get_matrix()
        
        assert result.shape == (10, 10)
        assert matrix._eval_mode is True

    def test_reparameterize(self):
        matrix = SVDSingleTransMatrix(size=10)
        
        matrix.reparameterize()
        
        assert matrix._eval_mode is True
        assert hasattr(matrix, 'matrix')
        assert hasattr(matrix, 'matrix_inv_t')

    def test_forward_basic(self):
        matrix = SVDSingleTransMatrix(size=10)
        inp = torch.randn(3, 10)
        
        result = matrix(inp)
        
        assert result.shape == inp.shape


class TestDiagonalTransMatrix:
    def test_init_default_params(self):
        matrix = DiagonalTransMatrix(size=10)
        
        assert matrix.size == 10
        assert hasattr(matrix, 'diag_scale')
        assert matrix.diag_scale.shape == (10,)

    def test_forward_normal(self):
        matrix = DiagonalTransMatrix(size=10)
        inp = torch.randn(3, 10)
        
        result = matrix(inp)
        
        assert result.shape == inp.shape

    def test_forward_with_inv_t(self):
        matrix = DiagonalTransMatrix(size=10)
        inp = torch.randn(3, 10)
        
        result = matrix(inp, inv_t=True)
        
        assert result.shape == inp.shape

    def test_forward_with_none_diag_scale(self):
        matrix = DiagonalTransMatrix(size=10)
        matrix.diag_scale = None
        inp = torch.randn(3, 10)
        
        result = matrix(inp)
        
        assert torch.equal(result, inp)

    def test_reparameterize(self):
        matrix = DiagonalTransMatrix(size=10)
        
        matrix.reparameterize()
        
        assert matrix.diag_scale is None

    def test_to_eval_mode(self):
        matrix = DiagonalTransMatrix(size=10)
        
        matrix.to_eval_mode()
        
        assert matrix.diag_scale is None

    def test_get_save_params(self):
        matrix = DiagonalTransMatrix(size=10)
        
        params = matrix.get_save_params()
        
        assert params == {}

