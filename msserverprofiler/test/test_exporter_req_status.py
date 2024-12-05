import unittest
from ms_server_profiler_analyze.exporters.exporter_req_status import ExporterReqStatus, ReqStatus


class TestExporterReqStatus(unittest.TestCase):
    def test_line_graph(self): 
        self.assertEqual(len(ReqStatus), 9)


if __name__ == '__main__':
    unittest.main()

