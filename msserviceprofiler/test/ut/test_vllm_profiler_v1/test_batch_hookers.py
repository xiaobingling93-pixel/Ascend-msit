import pytest
from unittest.mock import patch, MagicMock, call
import sys
import os
import threading
from collections import namedtuple, deque, Counter
import logging

# Revised ProfilerMock that properly tracks instance calls
class ProfilerMock:
    instance_calls = []

    def __init__(self, level=None):
        self.calls = []
        ProfilerMock.instance_calls.append(self.calls)

    def domain(self, name):
        self.calls.append(('domain', name))
        return self

    def res(self, res_id):
        self.calls.append(('res', res_id))
        return self

    def event(self, event_name):
        self.calls.append(('event', event_name))
        return self

    def metric(self, name, value):
        self.calls.append(('metric', name, value))
        return self

    def metric_scope(self, name, value):
        self.calls.append(('metric_scope', name, value))
        return self

    def metric_inc(self, name, value):
        self.calls.append(('metric_inc', name, value))
        return self

    def span_start(self, name):
        self.calls.append(('span_start', name))
        return self

    def span_end(self):
        self.calls.append('span_end')
        return self

    def attr(self, name, value):
        self.calls.append(('attr', name, value))
        return self

    @classmethod
    def reset(cls):
        cls.instance_calls = []

# Create fake modules in sys.modules
Level = namedtuple("Level", ["INFO"])("INFO")

sys.modules['ms_service_profiler'] = MagicMock()
sys.modules['ms_service_profiler'].Profiler = ProfilerMock
sys.modules['ms_service_profiler'].Level = Level

# Mock logger
logger = logging.getLogger("vllmProfiler")
logger.debug = MagicMock()

# Import test subject with environment variable
os.environ['VLLM_USE_V1'] = '-1'
from msserviceprofiler.vllm_profiler.vllm_v1 import batch_hookers

# Test utilities
def reset_thread_local():
    if hasattr(batch_hookers._thread_local, "hook_state"):
        del batch_hookers._thread_local.hook_state

def get_profiler_events():
    """Get all events from all Profiler instances"""
    events = []
    for calls in ProfilerMock.instance_calls:
        for call_item in calls:
            if call_item[0] == 'event':
                events.append(call_item[1])
    return events

def get_profiler_attrs():
    """Get all attributes from all Profiler instances"""
    attrs = []
    for calls in ProfilerMock.instance_calls:
        for call_item in calls:
            if call_item[0] == 'attr':
                attrs.append(call_item[1:])
    return attrs

def create_mock_request(request_id):
    req = MagicMock()
    req.request_id = request_id
    req.prompt_token_ids = [1, 2, 3]
    return req

def create_mock_scheduler_output(scheduled_new_reqs=None, scheduled_cached_reqs=None,
                                num_scheduled_tokens=None, finished_req_ids=None):
    output = MagicMock()
    output.scheduled_new_reqs = scheduled_new_reqs or []
    output.scheduled_cached_reqs = scheduled_cached_reqs or []
    output.num_scheduled_tokens = num_scheduled_tokens or {}
    output.finished_req_ids = finished_req_ids or set()
    return output

# Fixtures
@pytest.fixture(autouse=True, scope="function")
def cleanup():
    reset_thread_local()
    ProfilerMock.reset()
    logger.debug.reset_mock()
    yield
    reset_thread_local()
    ProfilerMock.reset()

# Tests
class TestCompareDeques:
    def test_compare_deques_given_overlapping_elements_when_compared_then_correct_diff(self):
        q1 = deque(['a', 'b', 'b', 'c'])
        q2 = deque(['b', 'c', 'd'])
        diff = batch_hookers.compare_deques(q1, q2)
        assert diff == Counter({'a': 1, 'b': 1, 'c': 0})

    def test_compare_deques_given_identical_queues_when_compared_then_empty_diff(self):
        q1 = deque(['a', 'b', 'c'])
        q2 = deque(['a', 'b', 'c'])
        diff = batch_hookers.compare_deques(q1, q2)
        assert diff == Counter()

    def test_compare_deques_given_empty_first_queue_when_compared_then_empty_diff(self):
        q1 = deque()
        q2 = deque(['a', 'b'])
        diff = batch_hookers.compare_deques(q1, q2)
        assert diff == Counter()

    def test_compare_deques_given_empty_second_queue_when_compared_then_full_diff(self):
        q1 = deque(['a', 'b'])
        q2 = deque()
        diff = batch_hookers.compare_deques(q1, q2)
        assert diff == Counter({'a': 1, 'b': 1})

class TestQueueProfiler:
    def test_queue_profiler_given_elements_removed_when_profiled_then_logs_dequeue(self):
        before = deque([MagicMock(request_id='req1'), MagicMock(request_id='req2')])
        after = deque([MagicMock(request_id='req1')])
        batch_hookers.queue_profiler(before, after, "test_queue")

        assert 'Dequeue' in get_profiler_events()

        # Verify queue metrics
        for calls in ProfilerMock.instance_calls:
            for call_item in calls:
                if call_item[0] == 'metric_scope' and call_item[1] == 'QueueName':
                    assert call_item[2] == "test_queue"
                if call_item[0] == 'metric' and call_item[1] == 'QueueSize':
                    assert call_item[2] == 1

    def test_queue_profiler_given_elements_added_when_profiled_then_logs_enqueue(self):
        before = deque([MagicMock(request_id='req1')])
        after = deque([MagicMock(request_id='req1'), MagicMock(request_id='req2')])
        batch_hookers.queue_profiler(before, after, "test_queue")

        assert 'Enqueue' in get_profiler_events()

        # Verify queue metrics
        for calls in ProfilerMock.instance_calls:
            for call_item in calls:
                if call_item[0] == 'metric_scope' and call_item[1] == 'QueueName':
                    assert call_item[2] == "test_queue"
                if call_item[0] == 'metric' and call_item[1] == 'QueueSize':
                    assert call_item[2] == 2

    def test_queue_profiler_given_multiple_changes_when_profiled_then_logs_both_events(self):
        before = deque([MagicMock(request_id='req1'), MagicMock(request_id='req2')])
        after = deque([MagicMock(request_id='req2'), MagicMock(request_id='req3')])
        batch_hookers.queue_profiler(before, after, "mixed_queue")

        events = get_profiler_events()
        assert 'Dequeue' in events
        assert 'Enqueue' in events

class TestHookState:
    def test_hook_state_given_new_instance_when_initialized_then_empty_containers(self):
        state = batch_hookers.HookState()
        assert state.request_id_to_prompt_token_len == {}
        assert state.request_id_to_iter_size == {}
        assert state.running == set()
        assert state.waiting == set()

class TestGetState:
    def test_get_state_given_thread_local_when_called_then_returns_same_instance(self):
        state1 = batch_hookers._get_state()
        state2 = batch_hookers._get_state()
        assert state1 is state2

    def test_get_state_given_different_threads_when_called_then_returns_different_instances(self):
        state_main = batch_hookers._get_state()
        state_thread = None

        def thread_func():
            nonlocal state_thread
            state_thread = batch_hookers._get_state()

        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join()

        assert state_main is not state_thread

class TestProcessInputs:
    def test_process_inputs_given_valid_request_when_processed_then_logs_event(self):
        mock_original = MagicMock(return_value="result")
        mock_this = MagicMock()

        result = batch_hookers.process_inputs(mock_original, mock_this, "req123")

        mock_original.assert_called_once_with(mock_this, "req123")
        assert result == "result"
        assert 'ReqState' in get_profiler_events()

        # Verify resource ID
        for calls in ProfilerMock.instance_calls:
            for call_item in calls:
                if call_item[0] == 'res':
                    assert call_item[1] == "req123"

class TestSchedule:
    def test_schedule_given_new_requests_when_processed_then_updates_state_and_logs(self):
        req1 = create_mock_request("req1")
        sched_output = create_mock_scheduler_output(
            scheduled_new_reqs=[req1],
            num_scheduled_tokens={"req1": 5}
        )
        mock_original = MagicMock(return_value=sched_output)
        mock_scheduler = MagicMock()
        mock_scheduler.running = deque([req1])
        mock_scheduler.waiting = deque()

        result = batch_hookers.schedule(mock_original, mock_scheduler)

        state = batch_hookers._get_state()
        assert state.request_id_to_iter_size["req1"] == 0

    def test_schedule_given_cached_requests_when_processed_then_updates_state(self):
        req1 = create_mock_request("req1")
        sched_output = create_mock_scheduler_output(
            scheduled_cached_reqs=[req1],
            num_scheduled_tokens={"req1": 5}
        )
        mock_original = MagicMock(return_value=sched_output)
        mock_scheduler = MagicMock()
        mock_scheduler.running = deque([req1])
        mock_scheduler.waiting = deque()

        # Prepopulate waiting state
        state = batch_hookers._get_state()
        state.waiting.add("req1")

        result = batch_hookers.schedule(mock_original, mock_scheduler)

    def test_schedule_given_preempted_requests_when_processed_then_updates_state(self):
        req1 = create_mock_request("req1")
        sched_output = create_mock_scheduler_output(num_scheduled_tokens={"req1": 5})
        mock_original = MagicMock(return_value=sched_output)
        mock_scheduler = MagicMock()
        mock_scheduler.running = deque()
        mock_scheduler.waiting = deque([req1])

        # Prepopulate running state
        state = batch_hookers._get_state()
        state.running.add("req1")

        result = batch_hookers.schedule(mock_original, mock_scheduler)

    def test_schedule_given_finished_requests_when_processed_then_cleans_state(self):
        req1 = create_mock_request("req1")
        sched_output = create_mock_scheduler_output(
            num_scheduled_tokens={"req1": 5},
            finished_req_ids={"req1"}
        )
        mock_original = MagicMock(return_value=sched_output)
        mock_scheduler = MagicMock()
        mock_scheduler.running = deque([req1])
        mock_scheduler.waiting = deque()

        # Prepopulate state
        state = batch_hookers._get_state()
        state.request_id_to_prompt_token_len["req1"] = 10
        state.request_id_to_iter_size["req1"] = 3

        result = batch_hookers.schedule(mock_original, mock_scheduler)

        assert "req1" not in state.request_id_to_prompt_token_len
        assert "req1" not in state.request_id_to_iter_size

    def test_schedule_given_prefill_batch_when_processed_then_sets_batch_type(self):
        req1 = create_mock_request("req1")
        sched_output = create_mock_scheduler_output(
            scheduled_new_reqs=[req1],
            num_scheduled_tokens={"req1": 5}
        )
        mock_original = MagicMock(return_value=sched_output)
        mock_scheduler = MagicMock()
        mock_scheduler.running = deque([req1])
        mock_scheduler.waiting = deque()

        batch_hookers.schedule(mock_original, mock_scheduler)

        attrs = get_profiler_attrs()
        assert ('batch_type', 'Prefill') in attrs

    def test_schedule_given_decode_batch_when_processed_then_sets_batch_type(self):
        req1 = create_mock_request("req1")
        sched_output = create_mock_scheduler_output(num_scheduled_tokens={"req1": 5})
        mock_original = MagicMock(return_value=sched_output)
        mock_scheduler = MagicMock()
        mock_scheduler.running = deque([req1])
        mock_scheduler.waiting = deque()

        # Set non-zero iteration
        state = batch_hookers._get_state()
        state.request_id_to_iter_size["req1"] = 1

        batch_hookers.schedule(mock_original, mock_scheduler)

        attrs = get_profiler_attrs()
        assert ('batch_type', 'Decode') in attrs

class TestFreeRequest:
    def test_free_request_given_running_request_when_freed_then_updates_state(self):
        mock_original = MagicMock()
        mock_this = MagicMock()
        req = create_mock_request("req1")
        req.status.name = "FINISHED"

        state = batch_hookers._get_state()
        state.running.add("req1")

        batch_hookers.free_request(mock_original, mock_this, req)

        assert "req1" not in state.running
        assert 'ReqState' in get_profiler_events()

        # Verify metrics
        for calls in ProfilerMock.instance_calls:
            if ('metric_inc', 'RUNNING', -1) in calls:
                assert ('metric_inc', 'FINISHED', 1) in calls

    def test_free_request_given_waiting_request_when_freed_then_updates_state(self):
        mock_original = MagicMock()
        mock_this = MagicMock()
        req = create_mock_request("req1")
        req.status.name = "FINISHED"

        state = batch_hookers._get_state()
        state.waiting.add("req1")

        batch_hookers.free_request(mock_original, mock_this, req)

        assert "req1" not in state.waiting
        assert 'ReqState' in get_profiler_events()

        # Verify metrics
        for calls in ProfilerMock.instance_calls:
            if ('metric_inc', 'WAITING', -1) in calls:
                assert ('metric_inc', 'FINISHED', 1) in calls

    def test_free_request_given_unknown_request_when_freed_then_no_state_change(self):
        mock_original = MagicMock()
        mock_this = MagicMock()
        req = create_mock_request("req1")
        req.status.name = "FINISHED"

        state = batch_hookers._get_state()
        batch_hookers.free_request(mock_original, mock_this, req)

        assert "req1" not in state.running
        assert "req1" not in state.waiting

class TestAddRequest:
    def test_add_request_given_new_request_when_added_then_updates_state(self):
        mock_original = MagicMock()
        mock_this = MagicMock()
        mock_this.waiting = deque([create_mock_request("req1")])
        req = create_mock_request("req1")

        batch_hookers.add_request(mock_original, mock_this, req)

        state = batch_hookers._get_state()
        assert "req1" in state.waiting
        assert len(state.waiting) == 1

        # Verify events
        events = get_profiler_events()
        assert 'ReqState' in events
        assert 'Enqueue' in events
