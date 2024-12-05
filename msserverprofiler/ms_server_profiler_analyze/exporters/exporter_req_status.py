from pathlib import Path
import pandas as pd
from enum import Enum   
from matplotlib import pyplot as plt

from ms_server_profiler.parse import ExporterBase
        

class ReqStatus(Enum):
    WAITING = 0
    PENDING = 1
    RUNNING = 2
    SWAPPED = 3
    RECOMPUTE = 4
    SUSPENDED = 5
    END = 6
    STOP = 7
    PREFILL_HOLD = 8

class ExporterReqStatus(ExporterBase):
    name = "req_status"

    @classmethod
    def intialize(cls, args):
        cls.args = args

    @classmethod
    def export(cls, data) -> None:
        df = data.get('tx_data_df')[['message', 'start_time', 'end_time', 'name']]

        
        counters = []
        cur = [0 for _ in range(len(ReqStatus))]
        for i, row in df.iterrows():
            message = row['message']
            message_dic = message

            name = message_dic.get('name', '')
            if name == 'httpReq':
                cur[0] += 1
            elif name =='ReqState':        
                ori_value = message_dic['ori_value']
                new_value = message_dic['new_value']
                cur[ori_value] -= 1
                cur[new_value] += 1            
            counters.append(cur.copy())
        
        
        counters = pd.DataFrame(data=counters, columns=[x.name for x in ReqStatus])
        counters['timestamp'] = df['start_time']
        counters.to_csv(Path(cls.args.output_path) / 'request_status.csv')
        
        plt.figure(figsize=(20, 6))
        for state in [x.name for x in ReqStatus]:
            plt.plot(counters['timestamp'], counters[state], label=state)
        plt.legend()
        plt.show()
        plt.savefig(Path(cls.args.output_path) / 'request_status.jpg')
