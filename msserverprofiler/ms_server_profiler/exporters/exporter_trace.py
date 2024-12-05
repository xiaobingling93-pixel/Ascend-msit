from ms_server_profiler.parse import ExporterBase


class ExporterTrace(ExporterBase):
    name = "trace"

    @classmethod
    def intialize(cls, args):
        cls.args = args

    @classmethod
    def export(cls, data) -> None:
        from parse_data_to_trace import create_trace_events, save_trace_data_into_json
        all_data_df, cpu_data_df = data['tx_data_df'], data['cpu_data_df']
        output = cls.args.output_path
        trace_data = create_trace_events(all_data_df, cpu_data_df)
        save_trace_data_into_json(trace_data, output)
