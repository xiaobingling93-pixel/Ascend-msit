import os
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
from msserviceprofiler.modelevalstate.config.config import Settings, OptimizerConfigField


class SGLangCommandConfig(BaseModel):
    port: str = ""
    model: str = ""
    device: str = ""
    others: str = ""


class SGLangConfig(BaseModel):
    output: Path = Path("sglang")
    process_name: str = "sglang"
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: SGLangCommandConfig = SGLangCommandConfig()
    target_field: List[OptimizerConfigField] = Field(default_factory=list)


class CusSettings(Settings):
    name: str = "sglang-inference-optimization"
    sglang: SGLangConfig = Field(default_factory=lambda data: SGLangConfig(output=data["output"].joinpath("sglang")),
                                 validate_default=True)
