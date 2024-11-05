import os 
import json

from dataclasses import dataclass
from msit_llm.transform import file_utils

@dataclass
class EnvVar:
    """
    环境变量
    """
    atb_speed_home_path: str = os.getenv("ATB_SPEED_HOME_PATH", None)



    def __post_init__(self):

        if self.atb_speed_home_path is not None:
            self.atb_speed_home_path = file_utils.standardize_path(self.atb_speed_home_path)
            file_utils.check_path_permission(self.atb_speed_home_path)

        
ENV = EnvVar()