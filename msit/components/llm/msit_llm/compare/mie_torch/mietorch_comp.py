import logging 
import os 
import csv 

import torch 

from msit_llm.common.log import logger 
from msit_llm.compare.utils.ge_dump_reader import GEDumpFileReader
from msit_llm.compare.utils.torch_dump_reader import TorchDumpFileReader
from msit_llm.compare.cmp_algorithm import CMP_ALG_MAP, CUSTOM_ALG_MAP
from components.utils.file_open_check import ms_open


class MIETorchCompare:
    def __init__(self, cpu_path: str, npu_path: str, json_path: str, output_path: str = "."):
        self.cpu_path = cpu_path
        self.npu_path = npu_path
        self.json_path = json_path
        self.output_path = output_path

        self.npu_reader = GEDumpFileReader(npu_path, json_path)
        self.cpu_reader = TorchDumpFileReader(cpu_path, json_path)
        self.cpu_keys = self.cpu_reader._get_keys()
        self.npu_keys = self.npu_reader._get_keys()
        self.output_path = output_path
    
    @staticmethod
    def check_tensor(golden_data_fp32, my_data_fp32):
        tensor_pass = True
        fail_reasons = []

        if len(golden_data_fp32) != len(my_data_fp32):
            fail_reasons.append("data shape doesn't match.")
            tensor_pass = False 
        if not torch.all(torch.isfinite(golden_data_fp32)):
            fail_reasons.append("cpu_data includes NAN or inf.")
            tensor_pass = False 
        if not torch.all(torch.isfinite(my_data_fp32)):
            fail_reasons.append("npu_data includes NAN or inf.")
            tensor_pass = False
        
        return tensor_pass, " ".join(fail_reasons)

    def compare(self):
        tensors = {}
        for cpu_key in self.cpu_keys:
            if cpu_key in self.npu_keys:
                cpu_tensor = self.cpu_reader.get_tensor(cpu_key)
                npu_tensor = self.npu_reader.get_tensor(cpu_key)
                tensors[cpu_key] = (cpu_tensor, npu_tensor)
        
        all_rows_data = []
        
        for key, (cpu_tensor, npu_tensor) in tensors.items():
            row_data = {"Key": self.cpu_reader.key_to_folder[key]}
            npu_tensor = torch.from_numpy(npu_tensor)
            cpu_tensor = cpu_tensor.reshape(-1).float()
            npu_tensor = npu_tensor.reshape(-1).float()

            tensor_pass, message = self.check_tensor(cpu_tensor, npu_tensor)

            if not tensor_pass:
                logger.debug(f"check_tensor failed: %s", message)
                row_data["cmp_fail_reason"] = message 
            else:
                fail_messages = []
                for name, cmp_func in list(CMP_ALG_MAP.items()) + list(CUSTOM_ALG_MAP.items()):
                    result, message = cmp_func(cpu_tensor, npu_tensor)
                    row_data[name] = result 
                    if len(message) > 0:
                        fail_messages.append(message)
                row_data["cmp_fail_reason"] = " ".join(fail_messages)

            all_rows_data.append(row_data)
        
        return self.save_compare_result_to_csv(all_rows_data)
    
    def save_compare_result_to_csv(self, all_rows_data: list) -> str:
        if not all_rows_data:
            logger.info("No data to save.")
            return "No data to save."
        
        sorted_rows = sorted(
            all_rows_data,
            key=lambda x: self.cpu_reader.key_to_id.get(x["Key"], float('inf'))
            )
        
        csv_file_path = os.path.join(self.output_path, 'comparison_results.csv')

        with ms_open(csv_file_path, mode="w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted_rows[0].keys())
            writer.writeheader()
            writer.writerows(sorted_rows)
        
        logger.info(f"Comparison results saved to %s", csv_file_path)

        return csv_file_path
        
