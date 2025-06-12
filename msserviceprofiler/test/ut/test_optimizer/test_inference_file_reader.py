from msserviceprofiler.modelevalstate.inference.file_reader import FileHanlder


class TestFileHandler:
    def test_load_static_data(self, static_file):
        fh = FileHanlder(static_file)
        fh.load_static_data()
        assert fh.hardware
        assert fh.env_info
        assert fh.mindie_info
        assert fh.model_config_info
        assert fh.model_struct_info
        assert fh.prefill_op_data
        assert fh.decode_op_data