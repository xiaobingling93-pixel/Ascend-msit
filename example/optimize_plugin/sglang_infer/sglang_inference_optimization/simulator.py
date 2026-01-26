import subprocess
from pydantic import BaseModel, Field
from typing import Optional, Tuple
from loguru import logger
from msserviceprofiler.modelevalstate.config.config import get_settings, OptimizerConfigField
from msserviceprofiler.modelevalstate.optimizer.interfaces.simulator import SimulatorInterface
from sglang_inference_optimization.settings import SGLangCommandConfig


class SGLangCommand:
    def __init__(self, command_config: SGLangCommandConfig):
        self.command_config = command_config

    @property
    def command(self):
        cmd = ["python", "-m", "sglang.launch_server",
                "--model-path", self.command_config.model,
                "--device", self.command_config.device,
                "--port", self.command_config.port,
                "--mem-fraction-static", "$MEM_FRACTION_STATIC",
                "--max-running-requests", "$MAX_RUNNING_REQUESTS",
                "--max-queued-requests", "$MAX_QUEUED_REQUESTS",
                "--max-prefill-tokens", "$MAX_PREFILL_TOKENS"]

        if self.command_config.others:
            cmd.extend(self.command_config.others.split())
        return cmd

class SGLangSimulator(SimulatorInterface):
    def __init__(self, config = None, *args, **kwargs):
        settings = get_settings()
        if settings.name != "sglang-inference-optimization":
            raise ValueError("Settings is invalidator.")
        self.config = settings.sglang
        super().__init__(*args, process_name=self.config.process_name, **kwargs)

        self.command = SGLangCommand(self.config.command).command

    @property
    def base_url(self) -> str:
        """
        获取服务的base url 属性
        Returns:

        """
        return f"http://127.0.0.1:{self.config.command.port}/health"

    def update_command(self):
        self.command = SGLangCommand(self.config.command).command

    def before_run(self, run_params: Optional[Tuple[OptimizerConfigField]] = None):
        self.update_command()
        super().before_run(run_params)
        subprocess.run(["pkill", "-KILL", "-f", "sglang"], stderr=subprocess.STDOUT, text=True)

    def stop(self, del_log: bool = True):
        """
        运行时，其他的准备工作。
        Returns:

        """
        try:
            subprocess.run(["pkill", "-KILL", "-f", "sglang"], stderr=subprocess.STDOUT, text=True)
        except subprocess.SubprocessError:
            pass
        super().stop(del_log)
