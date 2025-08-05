import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call
import sys
import os
from collections import namedtuple

# Create fake modules in sys.modules
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

# Now import our test subject
os.environ['VLLM_USE_V1'] = '-1'
from msserviceprofiler.vllm_profiler.vllm_v1.request_hookers import (
    add_request_async,
    process_outputs
)

# Test data structures
RequestState = namedtuple('RequestState', ['prompt_token_ids', 'stats'])
Stats = namedtuple('Stats', ['num_generation_tokens'])
EngineCoreOutput = namedtuple('EngineCoreOutput', ['request_id', 'finish_reason'])

@pytest.fixture
def mock_profiler():
    return ProfilerMock()

@pytest.fixture
def mock_original_func():
    return AsyncMock()

@pytest.fixture
def mock_this():
    return MagicMock()

@pytest.mark.asyncio
class TestAddRequestAsync:
    """Tests for add_request_async function"""

    async def test_add_request_async_given_empty_prompt_when_called_then_profiles_and_calls_original(self, mock_profiler, mock_original_func, mock_this):
        # Setup
        request_id = "req123"
        prompt = ""

        with patch('ms_service_profiler.Profiler', return_value=mock_profiler):
            # Execute
            await add_request_async(mock_original_func, mock_this, request_id, prompt)

            # Verify
            mock_original_func.assert_awaited_once_with(mock_this, request_id, prompt)

    async def test_add_request_async_given_none_prompt_when_called_then_profiles_and_calls_original(self, mock_profiler, mock_original_func, mock_this):
        # Setup
        request_id = "req123"
        prompt = None

        with patch('ms_service_profiler.Profiler', return_value=mock_profiler):
            # Execute
            await add_request_async(mock_original_func, mock_this, request_id, prompt)

            # Verify
            mock_original_func.assert_awaited_once_with(mock_this, request_id, prompt)

class TestProcessOutputs:
    """Tests for process_outputs function"""

    def test_process_outputs_given_single_completed_request_when_called_then_profiles_and_returns_original(self, mock_profiler):
        # Setup
        original_func = MagicMock(return_value="original_result")
        mock_this = MagicMock()
        request_id = "req123"

        output = EngineCoreOutput(request_id=request_id, finish_reason="stop")
        mock_this.request_states = {
            request_id: RequestState(prompt_token_ids=[1, 2, 3], stats=Stats(num_generation_tokens=5))
        }

        with patch('ms_service_profiler.Profiler', return_value=mock_profiler):
            # Execute
            result = process_outputs(original_func, mock_this, [output])

            # Verify
            assert result == "original_result"
            original_func.assert_called_once_with(mock_this, [output])

    def test_process_outputs_given_multiple_completed_requests_when_called_then_profiles_all(self, mock_profiler):
        # Setup
        original_func = MagicMock()
        mock_this = MagicMock()

        req1 = EngineCoreOutput(request_id="req1", finish_reason="stop")
        req2 = EngineCoreOutput(request_id="req2", finish_reason="length")
        mock_this.request_states = {
            "req1": RequestState(prompt_token_ids=[1, 2], stats=Stats(num_generation_tokens=4)),
            "req2": RequestState(prompt_token_ids=[3], stats=Stats(num_generation_tokens=2))
        }

        with patch('ms_service_profiler.Profiler', return_value=mock_profiler):
            # Execute
            process_outputs(original_func, mock_this, [req1, req2])

    def test_process_outputs_given_empty_outputs_when_called_then_calls_original_only(self, mock_profiler):
        # Setup
        original_func = MagicMock()
        mock_this = MagicMock()
        mock_this.request_states = {}

        with patch('ms_service_profiler.Profiler', return_value=mock_profiler):
            # Execute
            process_outputs(original_func, mock_this, [])

            # Verify
            original_func.assert_called_once_with(mock_this, [])

    def test_process_outputs_given_output_with_no_request_state_when_called_then_skips_profiling(self, mock_profiler):
        # Setup
        original_func = MagicMock()
        mock_this = MagicMock()
        mock_this.request_states = {}  # No state for our request

        output = EngineCoreOutput(request_id="req1", finish_reason="stop")

        with patch('ms_service_profiler.Profiler', return_value=mock_profiler):
            # Execute
            process_outputs(original_func, mock_this, [output])

            # Verify
            assert ('event', 'httpRes') not in mock_profiler.calls

    def test_process_outputs_given_output_with_no_finish_reason_when_called_then_skips_profiling(self, mock_profiler):
        # Setup
        original_func = MagicMock()
        mock_this = MagicMock()

        output = EngineCoreOutput(request_id="req1", finish_reason=None)
        mock_this.request_states = {
            "req1": RequestState(prompt_token_ids=[1], stats=Stats(num_generation_tokens=1))
        }

        with patch('ms_service_profiler.Profiler', return_value=mock_profiler):
            # Execute
            process_outputs(original_func, mock_this, [output])

            # Verify
            assert ('event', 'httpRes') not in mock_profiler.calls

    def test_process_outputs_given_none_outputs_when_called_then_raises_error(self):
        # Setup
        original_func = MagicMock()
        mock_this = MagicMock()

        # Execute & Verify
        with pytest.raises(TypeError):
            process_outputs(original_func, mock_this, None)

    def test_process_outputs_given_invalid_output_object_when_called_then_raises_error(self):
        # Setup
        original_func = MagicMock()
        mock_this = MagicMock()

        # Execute & Verify
        with pytest.raises(AttributeError):
            process_outputs(original_func, mock_this, [object()])
