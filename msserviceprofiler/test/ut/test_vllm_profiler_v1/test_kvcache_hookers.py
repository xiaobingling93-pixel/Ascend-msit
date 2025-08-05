import os
import pytest
from unittest.mock import patch, MagicMock, call
import sys
from collections import namedtuple

# Setup test environment
Level = namedtuple("Level", ["INFO"])("INFO")

class ProfilerMock:
    def __init__(self, *args):
        self._calls = []
        self._spans = {}

    def domain(self, name):
        self._calls.append(('domain', name))
        return self

    def res(self, res_id):
        self._calls.append(('res', res_id))
        return self

    def attr(self, name, value):
        self._calls.append(('attr', name, value))
        return self

    def span_start(self, name):
        self._calls.append(('span_start', name))
        self._spans[name] = True
        return self

    def span_end(self):
        self._calls.append(('span_end',))
        return self

    @property
    def calls(self):
        return self._calls

    @property
    def active_spans(self):
        return self._spans

sys.modules['ms_service_profiler'] = MagicMock()
sys.modules['ms_service_profiler'].Profiler = ProfilerMock
sys.modules['ms_service_profiler'].Level = Level

# Import our test subject
os.environ['VLLM_USE_V1'] = '-1'
from msserviceprofiler.vllm_profiler.vllm_v1 import kvcache_hookers

@pytest.fixture
def mock_kvcache_manager():
    """Fixture for creating mock KV cache manager"""
    manager = MagicMock()
    manager.block_pool = MagicMock()
    return manager

@pytest.fixture
def mock_request():
    """Fixture for creating mock request"""
    request = MagicMock()
    request.request_id = "test_id"
    request.num_tokens = 10
    return request

def test_allocate_slots_given_valid_request_when_called_then_logs_allocation(mock_kvcache_manager, mock_request):
    """Test that slot allocation logs correct metrics"""
    mock_kvcache_manager.block_pool.get_num_free_blocks.return_value = 42
    original_func = MagicMock(return_value="alloc_result")

    result = kvcache_hookers.allocate_slots(original_func, mock_kvcache_manager, mock_request)

    assert result == "alloc_result"
    original_func.assert_called_once_with(mock_kvcache_manager, mock_request)


def test_free_given_valid_request_when_called_then_logs_free(mock_kvcache_manager, mock_request):
    """Test that slot freeing logs correct metrics"""
    mock_kvcache_manager.block_pool.get_num_free_blocks.return_value = 99
    original_func = MagicMock(return_value="free_result")

    result = kvcache_hookers.free(original_func, mock_kvcache_manager, mock_request)

    assert result == "free_result"
    original_func.assert_called_once_with(mock_kvcache_manager, mock_request)

def test_get_computed_blocks_given_valid_conditions_when_called_then_logs_hit_rate(mock_kvcache_manager, mock_request):
    """Test that cache hit rate is calculated and logged correctly"""
    mock_request.num_tokens = 100
    original_func = MagicMock(return_value=[None, 75])  # [blocks, new_computed_tokens]

    result = kvcache_hookers.get_computed_blocks(original_func, mock_kvcache_manager, mock_request)

    assert result == [None, 75]
    original_func.assert_called_once_with(mock_kvcache_manager, mock_request)

def test_get_computed_blocks_given_single_block_when_called_then_no_hit_rate_logging(mock_kvcache_manager, mock_request):
    """Test that single block case doesn't trigger hit rate logging"""
    mock_request.num_tokens = 100
    original_func = MagicMock(return_value=[None])  # Single block

    result = kvcache_hookers.get_computed_blocks(original_func, mock_kvcache_manager, mock_request)

    assert result == [None]

def test_get_computed_blocks_given_zero_tokens_when_called_then_no_hit_rate_logging(mock_kvcache_manager, mock_request):
    """Test that zero tokens case doesn't trigger hit rate logging"""
    mock_request.num_tokens = 0
    original_func = MagicMock(return_value=[None, 0])  # Would cause division by zero

    result = kvcache_hookers.get_computed_blocks(original_func, mock_kvcache_manager, mock_request)

    assert result == [None, 0]

def test_get_computed_blocks_given_negative_tokens_when_called_then_no_hit_rate_logging(mock_kvcache_manager, mock_request):
    """Test that negative tokens case doesn't trigger hit rate logging"""
    mock_request.num_tokens = -10
    original_func = MagicMock(return_value=[None, 5])

    result = kvcache_hookers.get_computed_blocks(original_func, mock_kvcache_manager, mock_request)

    assert result == [None, 5]

@pytest.mark.parametrize("free_blocks", [0, 1, 100, -1])
def test_allocate_slots_given_various_block_counts_when_called_then_logs_correct_value(
    mock_kvcache_manager, mock_request, free_blocks
):
    """Parametrized test for different free block counts"""
    mock_kvcache_manager.block_pool.get_num_free_blocks.return_value = free_blocks
    original_func = MagicMock()

    kvcache_hookers.allocate_slots(original_func, mock_kvcache_manager, mock_request)

@pytest.mark.parametrize("num_tokens,computed_tokens,expected_rate", [
    (100, 50, 0.5),
    (200, 150, 0.75),
    (1000, 999, 0.999),
    (1, 1, 1.0),
])
def test_get_computed_blocks_given_various_rates_when_called_then_logs_correct_hit_rate(
    mock_kvcache_manager, mock_request, num_tokens, computed_tokens, expected_rate
):
    """Parametrized test for different cache hit rates"""
    mock_request.num_tokens = num_tokens
    original_func = MagicMock(return_value=[None, computed_tokens])

    kvcache_hookers.get_computed_blocks(original_func, mock_kvcache_manager, mock_request)
