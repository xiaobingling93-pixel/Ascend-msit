from collections import namedtuple

# Revised ProfilerMock that properly tracks instance calls
class Profiler:
    instance_calls = []

    def __init__(self, level=None):
        self.calls = []
        Profiler.instance_calls.append(self.calls)

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