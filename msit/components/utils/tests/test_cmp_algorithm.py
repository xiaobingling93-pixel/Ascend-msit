# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
import torch.nn.functional as F

from components.utils.cmp_algorithm import cosine_similarity, max_relative_error, mean_relative_error, \
    max_absolute_error, kl_divergence, mean_absolute_error, relative_euclidean_distance \


class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Test data preparation
        self.golden_data_all_zeros = torch.tensor([0.0, 0.0, 0.0])
        self.my_data_all_zeros = torch.tensor([0.0, 0.0, 0.0])
        self.golden_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        self.my_data = torch.tensor([1.1, 2.0, 3.1], dtype=torch.float64)
        self.golden_data_with_zero = torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64)
        self.my_data_with_zero = torch.tensor([0.0, 2.0, 3.1], dtype=torch.float64)

    def test_cosine_similarity(self):
        similarity, _ = cosine_similarity(self.golden_data, self.my_data)
        expected_similarity = torch.cosine_similarity(
            self.golden_data.double(), 
            self.my_data.double(),
            dim=0
        ).item()
        self.assertAlmostEqual(similarity, expected_similarity, places=10)
        similarity, _ = cosine_similarity(self.golden_data_all_zeros, self.my_data_all_zeros)
        self.assertEqual(similarity, 1.0)

    def test_max_relative_error(self):
        error, _ = max_relative_error(self.golden_data, self.my_data)
        expected_error = torch.max(torch.abs(self.my_data / self.golden_data - 1)).item()
        self.assertAlmostEqual(error, expected_error, places=6)

        error, _ = max_relative_error(self.golden_data_with_zero, self.my_data_with_zero)
        excepted = torch.max(torch.abs(self.my_data_with_zero[self.golden_data_with_zero != 0] / 
            self.golden_data_with_zero[self.golden_data_with_zero != 0] - 1)).item()
        self.assertAlmostEqual(error, excepted, places=6)

    def test_mean_relative_error(self):
        error, _ = mean_relative_error(self.golden_data, self.my_data)
        expected_error = torch.mean(torch.abs(self.my_data / self.golden_data - 1)).item()
        self.assertAlmostEqual(error, expected_error, places=6)

    def test_max_absolute_error(self):
        error, _ = max_absolute_error(self.golden_data, self.my_data)
        self.assertAlmostEqual(error, 0.1, places=6)

    def test_mean_absolute_error(self):
        error, _ = mean_absolute_error(self.golden_data, self.my_data)
        expected_error = torch.abs(self.my_data - self.golden_data).mean().item()
        self.assertAlmostEqual(error, expected_error, places=6)

    def test_kl_divergence(self):
        divergence, _ = kl_divergence(self.golden_data, self.my_data)
        expected_divergence = F.kl_div(
            F.log_softmax(self.my_data, dim=-1),
            F.softmax(self.golden_data, dim=-1),
            reduction="sum"
        ).item()
        self.assertAlmostEqual(divergence, expected_divergence, places=6)

    def test_relative_euclidean_distance(self):
        distance, _ = relative_euclidean_distance(self.golden_data, self.my_data)
        ground_truth_square_num = (self.golden_data ** 2).sum()
        expected_distance = ((self.my_data - self.golden_data) ** 2).sum() / ground_truth_square_num
        expected_distance = torch.sqrt(expected_distance).item()
        self.assertAlmostEqual(distance, expected_distance, places=6)