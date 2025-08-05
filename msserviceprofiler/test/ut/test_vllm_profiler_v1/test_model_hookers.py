import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call
import sys
import os
import threading
from collections import namedtuple
from contextlib import contextmanager

# Setup test environment
sys.modules['ms_service_profiler'] = MagicMock()
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

    def event(self, event_name):
        self._calls.append(('event', event_name))
        return self

    @property
    def calls(self):
        return self._calls

    @property
    def active_spans(self):
        return self._spans

sys.modules['ms_service_profiler'].Profiler = ProfilerMock
sys.modules['ms_service_profiler'].Level = Level

os.environ['VLLM_USE_V1'] = '-1'
from msserviceprofiler.vllm_profiler.vllm_v1 import model_hookers

@pytest.fixture
def reset_thread_local():
    """Fixture to reset thread-local state between tests"""
    if hasattr(model_hookers._thread_local, "hook_state"):
        del model_hookers._thread_local.hook_state
    yield
    if hasattr(model_hookers._thread_local, "hook_state"):
        del model_hookers._thread_local.hook_state

@pytest.fixture
def mock_scheduler_output():
    """Fixture for creating mock scheduler output"""
    def _make_mock_scheduler(
        new_reqs=None,
        cached_reqs=None,
        num_scheduled_tokens=None,
        finished_req_ids=None,
        total_num_scheduled_tokens=0
    ):
        mock_output = MagicMock()
        mock_output.scheduled_new_reqs = new_reqs or []
        mock_output.scheduled_cached_reqs = cached_reqs or []
        mock_output.num_scheduled_tokens = num_scheduled_tokens or {}
        mock_output.finished_req_ids = finished_req_ids or set()
        mock_output.total_num_scheduled_tokens = total_num_scheduled_tokens
        return mock_output
    return _make_mock_scheduler

def test_execute_model_given_empty_scheduler_output_when_called_then_no_profiling(reset_thread_local, mock_scheduler_output):
    """Test that empty scheduler output doesn't trigger profiling"""
    mock_output = mock_scheduler_output()
    original_func = MagicMock(return_value="test_result")

    result = model_hookers.execute_model(original_func, None, mock_output)

    assert result == "test_result"
    original_func.assert_called_once_with(None, mock_output)

def test_execute_model_given_new_requests_when_called_then_tracks_token_lengths(reset_thread_local, mock_scheduler_output):
    """Test that new requests have their prompt token lengths tracked"""
    Request = namedtuple("Request", ["req_id", "prompt_token_ids", "num_computed_tokens"])
    new_reqs = [
        Request(req_id=1, prompt_token_ids=[1,2,3], num_computed_tokens=3),
        Request(req_id=2, prompt_token_ids=[4,5,6,7], num_computed_tokens=0)
    ]
    mock_output = mock_scheduler_output(
        new_reqs=new_reqs,
        num_scheduled_tokens={1: 1, 2: 1},
        total_num_scheduled_tokens=2
    )
    original_func = MagicMock()

    model_hookers.execute_model(original_func, None, mock_output)

    state = model_hookers._get_state()
    assert state.request_id_to_prompt_token_len == {1: 3, 2: 4}

def test_execute_model_given_finished_requests_when_called_then_cleans_up_state(reset_thread_local, mock_scheduler_output):
    """Test that finished requests are cleaned up from state"""
    Request = namedtuple("Request", ["req_id", "prompt_token_ids", "num_computed_tokens"])
    new_reqs = [Request(req_id=1, prompt_token_ids=[1,2,3], num_computed_tokens=3)]
    mock_output = mock_scheduler_output(
        new_reqs=new_reqs,
        num_scheduled_tokens={1: 1},
        finished_req_ids={1},
        total_num_scheduled_tokens=1
    )
    original_func = MagicMock()

    # First call to populate state
    model_hookers.execute_model(original_func, None, mock_output)
    # Second call where request finishes
    model_hookers.execute_model(original_func, None, mock_output)

    state = model_hookers._get_state()
    assert 1 not in state.request_id_to_prompt_token_len
    assert 1 not in state.request_id_to_iter_size

def test_execute_model_given_multiple_calls_when_called_then_increments_iter_size(reset_thread_local, mock_scheduler_output):
    """Test that iteration counter increments with each call"""
    Request = namedtuple("Request", ["req_id", "prompt_token_ids", "num_computed_tokens"])
    new_reqs = [Request(req_id=1, prompt_token_ids=[1,2,3], num_computed_tokens=3)]
    mock_output = mock_scheduler_output(
        new_reqs=new_reqs,
        num_scheduled_tokens={1: 1},
        total_num_scheduled_tokens=1
    )
    original_func = MagicMock()

    # First call
    model_hookers.execute_model(original_func, None, mock_output)
    state = model_hookers._get_state()
    assert state.request_id_to_iter_size[1] == 0

    # Second call
    model_hookers.execute_model(original_func, None, mock_output)
    state = model_hookers._get_state()
    assert state.request_id_to_iter_size[1] == 1

def test_set_forward_context_given_active_profilers_when_called_then_manages_spans_correctly(reset_thread_local):
    """Test that context manager properly manages profiling spans"""
    original_func = MagicMock()
    original_func.return_value = MagicMock()
    original_func.return_value.__enter__ = MagicMock()
    original_func.return_value.__exit__ = MagicMock()

    # Setup state with active profilers
    state = model_hookers._get_state()
    state.preprocess_profiler = sys.modules['ms_service_profiler'].Profiler()
    state.forward_profiler = sys.modules['ms_service_profiler'].Profiler()
    state.postprocess_profiler = sys.modules['ms_service_profiler'].Profiler()

    with model_hookers.set_forward_context(original_func):
        pass

def test_set_forward_context_given_no_profilers_when_called_then_no_span_operations(reset_thread_local):
    """Test that context manager does nothing when no profilers are active"""
    original_func = MagicMock()
    original_func.return_value = MagicMock()
    original_func.return_value.__enter__ = MagicMock()
    original_func.return_value.__exit__ = MagicMock()

    # Ensure no profilers are active
    state = model_hookers._get_state()
    state.preprocess_profiler = None
    state.forward_profiler = None
    state.postprocess_profiler = None

    with model_hookers.set_forward_context(original_func):
        pass

# Parametrized tests for different batch scenarios
@pytest.mark.parametrize("new_reqs,cached_reqs,expected_type", [
    # New requests with computed_tokens < prompt length = prefill
    ([{"req_id": 1, "prompt_token_ids": [1,2,3], "num_computed_tokens": 0}], [], "Prefill"),
    # Cached requests with computed_tokens >= prompt length = decode
    ([], [{"req_id": 1, "prompt_token_ids": [1,2,3], "num_computed_tokens": 3}], "Decode"),
    # Mixed case should result in prefill
    (
        [{"req_id": 1, "prompt_token_ids": [1,2,3], "num_computed_tokens": 0}],
        [{"req_id": 2, "prompt_token_ids": [1,2,3], "num_computed_tokens": 3}],
        "Prefill"
    ),
])
def test_execute_model_given_various_batches_when_called_then_sets_correct_batch_type(
    reset_thread_local, mock_scheduler_output, new_reqs, cached_reqs, expected_type
):
    """Parametrized test for batch type detection"""
    Request = namedtuple("Request", ["req_id", "prompt_token_ids", "num_computed_tokens"])
    mock_new = [Request(**r) for r in new_reqs]
    mock_cached = [Request(**r) for r in cached_reqs]
    mock_output = mock_scheduler_output(
        new_reqs=mock_new,
        cached_reqs=mock_cached,
        num_scheduled_tokens={r.req_id: 1 for r in mock_new + mock_cached},
        total_num_scheduled_tokens=len(mock_new + mock_cached)
    )
    original_func = MagicMock()

    model_hookers.execute_model(original_func, None, mock_output)
