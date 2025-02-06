import csv
import os
import time
import queue
import threading
from threading import Lock
from collections import namedtuple

MAX_FILE_SIZE = 4 * 1024**3  # 4GB
_EVENT_TYPE = ["start", "start_process", "prefill", "decode", "finish", "error"]
EVENT_TYPE = namedtuple("event_type", _EVENT_TYPE)(*_EVENT_TYPE)


class ProfillingCSVWriter:
    def __init__(self, csv_file_pattern="profiling"):
        self.csv_file_pattern = csv_file_pattern
        self.request_stats, self.stats_lock, self.active_requests = {}, Lock(), 0
        self.current_csv_index, self.current_csv_file, self.current_file_size = 0, None, 0

        self.event_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.event_processors = {
            EVENT_TYPE.start: self._process_start_event,
            EVENT_TYPE.start_process: self._process_start_process_event,
            EVENT_TYPE.prefill: self._process_prefill_event,
            EVENT_TYPE.decode: self._process_decode_event,
            EVENT_TYPE.finish: self._process_finish_event,
            EVENT_TYPE.error: self._process_error_event,
        }

        thread = threading.Thread(target=self._background_writer, daemon=True)
        thread.start()

    def __del__(self):
        self.stop_event.set()

    def put_event(self, event):
        self.event_queue.put(event)

    def _get_next_filename(self):
        self.current_csv_index += 1
        return f"{self.csv_file_pattern}_{self.current_csv_index}.csv"

    def _write_to_csv(self, row):
        global current_csv_file, current_file_size
        if self.current_csv_file is None or self.current_file_size >= MAX_FILE_SIZE:
            if self.current_csv_file is not None:
                self.current_csv_file.close()
            filename = self._get_next_filename()
            self.current_csv_file = open(filename, "w", newline="")
            writer = csv.DictWriter(self.current_csv_file, fieldnames=row.keys())
            writer.writeheader()
            self.current_file_size = self.current_csv_file.tell()

        writer = csv.DictWriter(self.current_csv_file, fieldnames=row.keys())
        writer.writerow(row)
        self.current_csv_file.flush()
        self.current_file_size = self.current_csv_file.tell()

    def _process_start_event(self, http_rid, data, timestamp):
        self.request_stats[http_rid] = {
            "start_time": timestamp,
            "start_process_time": None,
            "queue_wait_time": None,
            "prefill_duration": 0,
            "decode_durations": [],
            "recv_tokens": 0,
            "reply_tokens": 0,
            "end_time": None,
            "error_type": None,
            "error_message": None,
            "concurrent_at_start": 0,
        }

    def _process_start_process_event(self, http_rid, data, timestamp):
        self.active_requests += 1
        self.request_stats[http_rid]["queue_wait_time"] = timestamp - self.request_stats[http_rid]["start_time"]
        self.request_stats[http_rid]["concurrent_at_start"] = self.active_requests

    def _process_prefill_event(self, http_rid, data, timestamp):
        self.request_stats[http_rid]["prefill_duration"] = data.get("duration", 0)
        self.request_stats[http_rid]["recv_tokens"] = data.get("recv_tokens", 0)

    def _process_decode_event(self, http_rid, data, timestamp):
        self.request_stats[http_rid]["decode_durations"].append(data.get("duration", 0))
        self.request_stats[http_rid]["reply_tokens"] += 1

    def _process_finish_event(self, http_rid, data, timestamp):
        self.active_requests -= 1
        stats = self.request_stats.pop(http_rid)
        total_decode_duration = sum(stats["decode_durations"])
        csv_row = {
            "http_rid": http_rid,
            "start_time": stats.get("start_time", 0),
            "execution_duration": timestamp - stats.get("start_time", 0),
            "queue_wait_time": stats["queue_wait_time"],
            "prefill_duration": stats["prefill_duration"],
            "total_decode_duration": total_decode_duration,
            "recv_tokens": stats["recv_tokens"],
            "reply_tokens": stats["reply_tokens"],
            "error_type": stats["error_type"],
            "error_message": stats["error_message"],
            "concurrent_at_start": stats["concurrent_at_start"],
        }
        self._write_to_csv(csv_row)

    def _process_error_event(self, http_rid, data, timestamp):
        self.active_requests -= 1
        stats = self.request_stats.pop(http_rid)
        stats["error_type"] = data.get("error_type")
        stats["error_message"] = data.get("error_message")
        csv_row = {
            "http_rid": http_rid,
            "start_time": stats.get("start_time", 0),
            "execution_duration": timestamp - stats.get("start_time", 0),
            "queue_wait_time": stats.get("queue_wait_time", 0),
            "prefill_duration": stats.get("prefill_duration", 0),
            "total_decode_duration": sum(stats.get("decode_durations", [])),
            "recv_tokens": stats.get("recv_tokens", 0),
            "reply_tokens": stats.get("reply_tokens", 0),
            "error_type": stats["error_type"],
            "error_message": stats["error_message"],
            "concurrent_at_start": stats.get("concurrent_at_start", 0),
        }
        _write_to_csv(csv_row)

    def _process_event(self, event):
        event_type = event.get("event_type", None)
        http_rid = event.get("http_rid", None)
        data = event.get("data", {})
        timestamp = event.get("timestamp", None)
        if event_type not in self.event_processors:
            return  # Unknown event_type
        if event_type != EVENT_TYPE.start and http_rid not in self.request_stats:
            return  # http_rid not started yet

        with self.stats_lock:
            self.event_processors[event_type](http_rid, data, timestamp)

    def _background_writer(self):
        while not self.stop_event.is_set() or not self.event_queue.empty():
            try:
                event = self.event_queue.get(timeout=1)
                self._process_event(event)
            except queue.Empty:
                continue
        if self.current_csv_file:
            self.current_csv_file.close()


profiling_csv_writer = ProfillingCSVWriter()
